"""
nav.py - odometry, pose filter and goal-seeking for the tag-localized robot.

Motion model (truth, dead reckoning and filter all use this form):
    turn by  k * turn + c (c only on walking steps),  then walk  s * fwd
  s : step-length scale (actual / commanded)
  k : turn scale        (actual / commanded)
  c : extra turn per walking step (curve)

Motors      : simulation only - the robot's true, unknown s, k, c, its random
              slip, and (optionally) slow random-walk drift of s, k, c.
OdometryEKF : pose filter [x, y, yaw].  With calibrate=True the state also
              holds s, k, c, which are learned online from the tag fixes
              (augmented-state EKF); otherwise the filter assumes the motors
              do what they are told and covers the difference with large
              process noise.
Navigator   : go-to-goal policy that turns to look at a landmark whenever
              the position uncertainty grows too large.
OdometryEKF and Navigator are robot-side code, like the estimators in tagsim.
"""
from dataclasses import dataclass
import numpy as np
from tagsim import wrap


def motion(x, fwd, turn):
    """Turn-then-walk motion model used for truth, dead reckoning and EKF."""
    th = x[2] + turn
    return np.array([x[0] + fwd * np.cos(th), x[1] + fwd * np.sin(th),
                     wrap(th)])


# ----------------------------------------------------------------------------
# Simulation: the robot's true motor behaviour
# ----------------------------------------------------------------------------
DRIFT = (0.002, 0.002, 0.03)     # default drift sd per step: s, k, c (deg)


@dataclass
class Motors:
    """The robot's true (unknown) motor personality.  Defaults are the
    robot of experiment C: walks 18 % short, under-turns 10 %, curves left."""
    scale: float = 0.82          # s
    turn_scale: float = 0.9      # k
    curve_deg: float = 1.5       # c (deg per walking step)
    slip: float = 0.08           # random step-length error (fraction, 1 sd)
    turn_noise_deg: float = 2.0  # random heading error per step (deg, 1 sd)
    drift: tuple = (0.0, 0.0, 0.0)   # random-walk sd per step of s, k, c (deg)

    @property
    def biases(self):
        return np.array([self.scale, self.turn_scale, np.radians(self.curve_deg)])

    def move(self, x, fwd, turn, rng, drift_rng=None):
        """True motion for one command; then s, k, c drift a little.
        Drift draws come from drift_rng so they never disturb rng."""
        x = motion(x, fwd * self.scale * (1 + rng.normal(0, self.slip)),
                   turn * self.turn_scale + np.radians(self.curve_deg) * (fwd > 0)
                   + rng.normal(0, np.radians(self.turn_noise_deg)))
        if drift_rng is not None and any(self.drift):
            ds, dk, dc = self.drift
            self.scale += drift_rng.normal(0, ds)
            self.turn_scale += drift_rng.normal(0, dk)
            self.curve_deg += drift_rng.normal(0, dc)
        return x


def random_motors(rng, drift=DRIFT):
    """A random robot: each one walks, turns and curves differently."""
    return Motors(scale=rng.uniform(0.8, 1.1), turn_scale=rng.uniform(0.85, 1.1),
                  curve_deg=rng.uniform(-2.0, 2.0), drift=drift)


# ----------------------------------------------------------------------------
# Pose filter (runs on the robot)
# ----------------------------------------------------------------------------
def fix_cov(nearest_dist):
    """Covariance of a tag fix (x, y, yaw): trust nearby tags more."""
    sd = 0.01 + 0.012 * nearest_dist
    return np.diag([sd ** 2, sd ** 2, np.radians(0.3 + 0.5 * nearest_dist) ** 2])


class OdometryEKF:
    """EKF over [x, y, yaw] (+ [s, k, c] when calibrate=True).

    Uncalibrated: predicts with s = k = 1, c = 0 and a large process noise
    (q_pos of the step, q_yaw per step) to cover the unknown motor errors.
    Calibrated: s, k, c are state, so q_pos / q_yaw only need to cover the
    random slip; bias_drift lets the learned values follow slow changes.
    Fixes measure [x, y, yaw]; a fix more than gate (chi^2) away from the
    prediction is rejected as an outlier.
    """

    def __init__(self, x0, calibrate=False, q_pos=None, q_yaw_deg=None,
                 bias0=(1.0, 1.0, 0.0), bias_sd=(0.15, 0.15, np.radians(2.5)),
                 bias_drift=(0.003, 0.003, np.radians(0.05)), gate=16.0):
        self.calibrate = calibrate
        self.q_pos = q_pos if q_pos is not None else (0.10 if calibrate else 0.25)
        self.q_yaw = np.radians(q_yaw_deg if q_yaw_deg is not None
                                else (2.5 if calibrate else 4.0))
        self.gate = gate
        self.x = np.asarray(x0, float).copy()
        self.P = np.diag([1e-4, 1e-4, 1e-4])
        if calibrate:
            self.x = np.concatenate([self.x, bias0])
            self.P = np.diag(np.concatenate([np.diag(self.P),
                                             np.square(bias_sd)]))
            self.Qb = np.square(bias_drift)
        n = len(self.x)
        self.H = np.eye(3, n)

    @property
    def pose(self):
        return self.x[:3]

    @property
    def biases(self):
        """Current belief about (s, k, c); nominal if not calibrating."""
        return self.x[3:] if self.calibrate else np.array([1.0, 1.0, 0.0])

    @property
    def pos_sd(self):
        """Position uncertainty: sd along the worst direction (m)."""
        return np.sqrt(np.linalg.eigvalsh(self.P[:2, :2]).max())

    def predict(self, fwd, turn):
        s, k, c = self.biases
        walk = float(fwd > 0)
        d = s * fwd
        th = self.x[2] + k * turn + c * walk
        sn, cs = np.sin(th), np.cos(th)
        F = np.eye(len(self.x))
        F[0, 2], F[1, 2] = -d * sn, d * cs
        if self.calibrate:
            F[0, 3], F[1, 3] = fwd * cs, fwd * sn
            F[0, 4], F[1, 4], F[2, 4] = -d * sn * turn, d * cs * turn, turn
            F[0, 5], F[1, 5], F[2, 5] = -d * sn * walk, d * cs * walk, walk
        self.x[:3] = motion(self.x, d, k * turn + c * walk)
        q = [(self.q_pos * fwd) ** 2 + 1e-6, (self.q_pos * fwd) ** 2 + 1e-6,
             self.q_yaw ** 2]
        if self.calibrate:
            q += list(self.Qb)
        self.P = F @ self.P @ F.T + np.diag(q)

    def update(self, z, R):
        """Fuse a fix z = (x, y, yaw); returns True if accepted."""
        H = self.H
        y = np.asarray(z, float) - self.x[:3]
        y[2] = wrap(y[2])
        S = H @ self.P @ H.T + R
        if y @ np.linalg.solve(S, y) >= self.gate:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        return True


# ----------------------------------------------------------------------------
# Goal seeking with active looking (runs on the robot)
# ----------------------------------------------------------------------------
@dataclass
class Landmark:
    xy: np.ndarray           # map position
    normal: np.ndarray       # direction it faces (2D), None = visible all round
    max_range: float         # beyond this it is unlikely to be detected (m)

    def in_view_cone(self, p, max_off_deg=60.0):
        if self.normal is None:
            return True
        v = p - self.xy
        return v @ self.normal > np.cos(np.radians(max_off_deg)) * np.linalg.norm(v)


class Navigator:
    """Walk to a goal; look at a landmark when lost-ish.

    go     : turn towards the goal (max_turn per step), walk up to `step`
             once roughly facing it.  Arrived when the estimate is within
             arrive_m of the goal and the position sd is below look_sd.
    look   : triggered when the position sd exceeds look_sd (or on arrival
             without a recent fix).  Turn to the most promising landmark
             (in range and facing us, else nearest) and wait up to
             `patience` photos for a fix.
    search : no fix while facing it - spin on the spot for one full turn.
             If that fails too, give up looking for `retry` steps and carry
             on blind (arrival is then allowed without a fix).
    Any accepted fix ends a look / search.
    look_only_in_range: when no landmark is within its max_range (and
             facing us), don't look at all - carry on blind for `retry`
             steps instead of a futile look / search.
    """

    def __init__(self, landmarks, look_sd=0.05, arrive_m=0.03, step=0.10,
                 max_turn_deg=30.0, patience=3, retry=8, look_only_in_range=False):
        self.landmarks = landmarks
        self.look_sd, self.arrive_m, self.step = look_sd, arrive_m, step
        self.max_turn = np.radians(max_turn_deg)
        self.patience, self.retry = patience, retry
        self.look_only_in_range = look_only_in_range
        self.stats = {"looks": 0, "searches": 0, "gave_up": 0, "out_of_range": 0}

    def set_goal(self, goal):
        self.goal = np.asarray(goal, float)
        self.mode, self.blind_left = "go", 0

    def _turn_to(self, x, target):
        b = wrap(np.arctan2(target[1] - x[1], target[0] - x[0]) - x[2])
        return b, float(np.clip(b, -self.max_turn, self.max_turn))

    def _start_look(self, x):
        p = x[:2]
        good = [l for l in self.landmarks if l.in_view_cone(p)
                and np.linalg.norm(l.xy - p) < l.max_range]
        if not good and self.look_only_in_range:
            self.blind_left = self.retry
            self.stats["out_of_range"] += 1
            return False
        pool = good or self.landmarks
        self.target = min(pool, key=lambda l: np.linalg.norm(l.xy - p))
        self.mode, self.facing = "look", 0
        self.stats["looks"] += 1
        return True

    def command(self, x, pos_sd, fixed):
        """Next (fwd, turn), or None when the goal is reached.
        x, pos_sd: current estimate; fixed: last photo gave an accepted fix."""
        if fixed:
            self.blind_left = 0
            if self.mode != "go":
                self.mode = "go"
        if self.blind_left > 0:
            self.blind_left -= 1
        if self.mode == "go":
            dist = np.linalg.norm(self.goal - x[:2])
            lost = pos_sd > self.look_sd and self.blind_left == 0
            if lost and not self._start_look(x):
                lost = False                  # nothing in range: go on blind
            if self.mode == "go":
                if dist < self.arrive_m and not lost:
                    return None
                b, turn = self._turn_to(x, self.goal)
                fwd = min(self.step, dist) if abs(b) < self.max_turn else 0.0
                return fwd, turn
        if self.mode == "look":
            b, turn = self._turn_to(x, self.target.xy)
            if abs(b) < np.radians(8):
                self.facing += 1
                if self.facing > self.patience:
                    self.mode, self.search_left = "search", 12
                    self.stats["searches"] += 1
            if self.mode == "look":
                return 0.0, turn
        # search: one full turn on the spot
        self.search_left -= 1
        if self.search_left <= 0:
            self.mode, self.blind_left = "go", self.retry
            self.stats["gave_up"] += 1
        return 0.0, self.max_turn
