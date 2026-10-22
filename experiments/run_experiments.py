"""
run_experiments.py - accuracy experiments for tagsim.py

  python3 -m experiments.run_experiments [output_dir]    (from the repo root;
                                    default output: results/localization)

A. Monte Carlo over random robot poses in a 4 x 3 m arena under realistic
   conditions (blur, noise, JPEG, body bobbing, 1 cm / 1 deg tag-placement
   error, slightly-off camera calibration).  Error vs distance per method.
B. Sensitivity: one factor changed at a time (tag size, spacing, motion
   blur, calibration, map error, camera resolution).
C. A walking trajectory with badly-calibrated motor odometry, with and
   without tag fixes fused in a small EKF.
Also saves an annotated example camera frame.
"""
import sys, os, json, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from tagsim import *
from nav import motion, Motors, OdometryEKF, fix_cov

OUT = "results/localization"  # set from the command line in __main__

# Chart styling (reference data-viz palette, light mode)
INK, INK2, MUTED, GRID, AXIS, SURF = ("#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7", "#fcfcfb")
C = {"single": "#2a78d6", "multi": "#eb6834", "floor": "#1baf7a",
     "floor_rigid": "#eda100"}
LABEL = {"single": "One tag (IPPE)", "multi": "All tags, 6-DOF PnP",
         "floor": "All tags + floor constraint", "floor_rigid":
         "Floor constraint, camera assumed rigid"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": INK2})

METHODS = ["single", "multi", "floor_rigid", "floor"]


def plain_log(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))


def random_pose(rng, W, D, margin=0.3):
    return (rng.uniform(margin, W - margin), rng.uniform(margin, D - margin),
            rng.uniform(-np.pi, np.pi))


def realistic_setup(rng, tag_size=0.15, spacing=0.75, map_pos=0.01,
                    map_yaw=1.0, f_err=0.01, c_err=3.0, cam=None,
                    room=(4.0, 3.0), tag_height=0.20):
    tags, (W, D) = make_arena(room[0], room[1], tag_size=tag_size,
                              tag_height=tag_height, spacing=spacing)
    cam = cam or Camera()
    map_tags = perturb_map(tags, rng, map_pos, map_yaw) if map_pos > 0 else None
    K_est = perturb_intrinsics(cam.K, rng, f_err, c_err)
    return tags, map_tags, cam, K_est, (W, D)


def summarise(results, methods=METHODS):
    n = len(results)
    hit = [r for r in results if r.n_tags > 0]
    s = {"frames": n, "coverage_pct": 100 * len(hit) / n,
         "mean_tags_seen": float(np.mean([r.n_tags for r in results]))}
    for m in methods:
        e = np.array([r.err[m] for r in hit])
        if len(e) == 0:
            continue
        s[m] = {"pos_median_cm": 100 * np.median(e[:, 0]),
                "pos_p90_cm": 100 * np.percentile(e[:, 0], 90),
                "pos_p99_cm": 100 * np.percentile(e[:, 0], 99),
                "yaw_median_deg": np.median(e[:, 1]),
                "yaw_p90_deg": np.percentile(e[:, 1], 90),
                "gross_fail_pct": 100 * np.mean(e[:, 0] > 0.25)}
    return s


# ---------------------------------------------------------------------------
# Example frame
# ---------------------------------------------------------------------------
def example_frame():
    rng = np.random.default_rng(7)
    tags, map_tags, cam, K_est, _ = realistic_setup(rng)
    pose = (1.3, 1.1, np.radians(-55))
    r, img, det = measure(pose, tags, cam, Degradation(), rng,
                          K_est=K_est, map_tags=map_tags, return_image=True)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for tag, uv in det:
        cv2.polylines(vis, [uv.astype(np.int32)], True, (214, 120, 42), 2)
        _, rv, tv = cv2.solvePnP(tag.corners_tag(), uv, cam.K, None,
                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
        cv2.drawFrameAxes(vis, cam.K, None, rv, tv, tag.size * 0.6, 2)
        cv2.putText(vis, f"id {tag.id}", tuple(uv[0].astype(int) + [0, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (214, 120, 42), 1, cv2.LINE_AA)
    e = r.est["floor"]
    txt = (f"true  x={pose[0]:.3f} y={pose[1]:.3f} yaw={np.degrees(pose[2]):.1f}")
    txt2 = (f"est   x={e[0]:.3f} y={e[1]:.3f} yaw={np.degrees(e[2]):.1f}")
    for i, t in enumerate((txt, txt2)):
        cv2.putText(vis, t, (10, 450 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, "example_frame.png"), vis)


# ---------------------------------------------------------------------------
# A. Main Monte Carlo
# ---------------------------------------------------------------------------
def experiment_A(N=1200):
    rng = np.random.default_rng(42)
    tags, map_tags, cam, K_est, (W, D) = realistic_setup(rng)
    det = make_detector()
    res = [measure(random_pose(rng, W, D), tags, cam, Degradation(), rng,
                   detector=det, K_est=K_est, map_tags=map_tags)
           for _ in range(N)]

    hit = [r for r in res if r.n_tags > 0]
    d = np.array([r.nearest_dist for r in hit])
    bins = np.arange(0.25, 4.01, 0.35)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k, unit, title in ((axes[0], 0, 100, "Position error (cm)"),
                               (axes[1], 1, 1, "Heading error (deg)")):
        for m in METHODS:
            e = np.array([r.err[m][k] for r in hit]) * unit
            med = [np.median(e[(d >= a) & (d < b)]) if np.sum((d >= a) & (d < b)) >= 15
                   else np.nan for a, b in zip(bins[:-1], bins[1:])]
            ax.plot(centers, med, color=C[m], lw=2, marker="o", ms=4,
                    label=LABEL[m])
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlabel("Distance to nearest detected tag (m)")
        ax.set_title(f"Median {title.lower()}", loc="left")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Single-camera tag localization: error vs distance "
                 f"({len(hit)} simulated frames)", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_vs_distance.png"), dpi=150)
    plt.close(fig)
    return summarise(res)


# ---------------------------------------------------------------------------
# B. Sensitivity
# ---------------------------------------------------------------------------
def experiment_B(N=300):
    scenarios = {
        "baseline (15 cm tags, every 0.75 m, 640x480)": {},
        "smaller tags (10 cm)": {"tag_size": 0.10},
        "larger tags (20 cm)": {"tag_size": 0.20},
        "sparse tags (every 1.5 m)": {"spacing": 1.5},
        "photo while walking (5 px motion blur)": {"deg": Degradation(motion_px=5)},
        "uncalibrated camera (f off 8%, centre off 15 px)": {"f_err": 0.08, "c_err": 15},
        "sloppy tag map (3 cm, 3 deg)": {"map_pos": 0.03, "map_yaw": 3.0},
        "low-res camera (320x240)": {"cam": Camera(width=320, height=240)},
    }
    out = {}
    for name, kw in scenarios.items():
        rng = np.random.default_rng(123)
        deg = kw.pop("deg", Degradation())
        tags, map_tags, cam, K_est, (W, D) = realistic_setup(rng, **kw)
        det = make_detector()
        res = [measure(random_pose(rng, W, D), tags, cam, deg, rng,
                       detector=det, K_est=K_est, map_tags=map_tags)
               for _ in range(N)]
        out[name] = summarise(res, ["single", "floor"])
        f = out[name].get("floor", {})
        print(f"  {name:52s} coverage {out[name]['coverage_pct']:5.1f}%  "
              f"floor median {f.get('pos_median_cm', np.nan):5.1f} cm  "
              f"p90 {f.get('pos_p90_cm', np.nan):5.1f} cm", flush=True)
    return out


# ---------------------------------------------------------------------------
# C. Trajectory: bad odometry + tag fixes in an EKF
# ---------------------------------------------------------------------------
def simulate_run(steps=240, on_step=None, seed=5, cube=None, waypoints=None,
                 look_at_cube=False, motors=None, calibrate=False, drift_seed=None):
    """Walk the robot round a waypoint loop with badly-calibrated motors.

    motors    : the robot's true motor behaviour (default Motors(): 18 % short
                steps, 10 % under-turn, 1.5 deg/step curve, no drift).  It is
                copied, so drift inside the run never leaks out.
    calibrate : the EKF also learns s, k, c online (nav.OdometryEKF).
    drift_seed: seed for the motors' drift (default: seed + 1000).

    Each step: plan a command from the fused estimate, move (truth, dead
    reckoning, EKF prediction), stop, take a photo, fuse any tag fix.

    With a cube: every photo that shows it places the cube on the map
    (floor-constrained relative pose composed with the fused robot pose)
    and a CubeTracker fuses those placements.  With look_at_cube, once the
    cube has been seen the robot turns to face it at every waypoint and
    takes a few photos before walking on.

    on_step(info) is called after every step with a dict holding the
    states before/after the step, the photo, detections and the fix.
    It must not use the simulation's random generator, so recording a
    video never changes the numbers.
    Returns (log, ctx): per-step arrays and the scene used.
    """
    rng = np.random.default_rng(seed)
    tags, map_tags, cam, K_est, (W, D) = realistic_setup(rng)
    det = make_detector()
    # Standing still for the photo: little bobbing, no motion blur.
    deg = Degradation(bob_pitch_deg=0.7, bob_roll_deg=0.5, bob_dz_m=0.002)

    # This robot's (unknown) motor personality.
    motors = Motors(**vars(motors)) if motors is not None else Motors()
    drift_rng = np.random.default_rng(seed + 1000 if drift_seed is None
                                      else drift_seed)

    waypoints = waypoints or [(3.3, 0.7), (3.3, 2.3), (0.7, 2.3), (0.7, 0.7)]
    x_true = np.array([0.7, 0.7, 0.0])
    x_dr = x_true.copy()
    ekf = OdometryEKF(x_true, calibrate=calibrate)
    x_kf = ekf.pose.copy()
    wp = 0
    log = {"true": [], "dr": [], "kf": [], "fix": [], "n_tags": [],
           "bias_true": [], "bias_est": [], "bias_sd": []}
    ctx = {"tags": tags, "map_tags": map_tags, "cam": cam, "K_est": K_est,
           "W": W, "D": D, "steps": steps, "start": x_true.copy(), "cube": cube}
    if cube is not None:
        tracker, look_left = CubeTracker(), 0
        ctx["cube_tracker"] = tracker
        for key in ("cube_in_frame", "cube_faces", "cube_dist", "cube_T",
                    "cube_accepted",
                    "cube_err_world", "cube_err_rel", "cube_err_fused"):
            log[key] = []

    for k in range(steps):
        prev = {"true": x_true.copy(), "dr": x_dr.copy(), "kf": x_kf.copy()}
        # Planner uses the fused estimate to pick its command.
        tx, ty = waypoints[wp % len(waypoints)]
        if np.hypot(tx - x_kf[0], ty - x_kf[1]) < 0.15:
            wp += 1
            tx, ty = waypoints[wp % len(waypoints)]
            if look_at_cube and tracker.T is not None:
                look_left = 3
        if look_at_cube and look_left > 0:
            # Turn on the spot towards where the cube is believed to be.
            cx, cy = tracker.T[:2, 3]
            bearing = wrap(np.arctan2(cy - x_kf[1], cx - x_kf[0]) - x_kf[2])
            turn = np.clip(bearing, -np.radians(30), np.radians(30))
            fwd = 0.0
            if abs(bearing) < np.radians(8):
                look_left -= 1
        else:
            bearing = wrap(np.arctan2(ty - x_kf[1], tx - x_kf[0]) - x_kf[2])
            turn = np.clip(bearing, -np.radians(30), np.radians(30))
            fwd = 0.10 if abs(bearing) < np.radians(30) else 0.0

        # Truth: what the motors actually do.
        x_true = motors.move(x_true, fwd, turn, rng, drift_rng)
        x_true[:2] = np.clip(x_true[:2], 0.25, [W - 0.25, D - 0.25])

        # Dead reckoning: integrate commands only.
        x_dr = motion(x_dr, fwd, turn)
        ekf.predict(fwd, turn)
        kf_pred = ekf.pose.copy()

        # Photo + tag measurement.
        r, img, dets = measure(tuple(x_true), tags, cam, deg, rng, detector=det,
                               K_est=K_est, map_tags=map_tags, return_image=True,
                               cube=cube)
        got = r.n_tags > 0 and ekf.update(np.array(r.est["floor"]),
                                          fix_cov(r.nearest_dist))
        x_kf = ekf.pose.copy()
        for key, v in (("true", x_true), ("dr", x_dr), ("kf", x_kf)):
            log[key].append(v.copy())
        log["fix"].append(got)
        log["n_tags"].append(r.n_tags)
        log["bias_true"].append(motors.biases)
        log["bias_est"].append(ekf.biases.copy())
        log["bias_sd"].append(np.sqrt(np.diag(ekf.P)[3:]) if calibrate
                              else np.zeros(3))
        if cube is not None:
            T_cube, accepted = None, False
            if r.cube_faces:
                T_rel = r.cube_rel["floor"]
                T_cube = robot_T(*x_kf) @ T_rel
                accepted = tracker.add(T_cube, np.linalg.norm(T_rel[:2, 3]))
            nan2 = (np.nan, np.nan)
            log["cube_in_frame"].append(r.cube_in_frame)
            log["cube_faces"].append(r.cube_faces)
            log["cube_dist"].append(r.cube_dist)
            log["cube_T"].append(T_cube)
            log["cube_accepted"].append(accepted)
            log["cube_err_world"].append(
                cube_error(T_cube, cube.T) if T_cube is not None else nan2)
            log["cube_err_rel"].append(r.cube_err_rel.get("floor", nan2))
            log["cube_err_fused"].append(
                cube_error(tracker.T, cube.T) if tracker.T is not None else nan2)
        if on_step is not None:
            on_step({"k": k, "prev": prev, "kf_pred": kf_pred,
                     "new": {"true": x_true.copy(), "dr": x_dr.copy(),
                             "kf": x_kf.copy()},
                     "cmd": (fwd, turn), "image": img, "dets": dets,
                     "result": r, "fix": got, "waypoint": (tx, ty),
                     "cube_fused": tracker.T if cube is not None else None})

    log = {k: v if k == "cube_T" else np.array(v) for k, v in log.items()}
    return log, ctx


def experiment_C(steps=240):
    log, ctx = simulate_run(steps)
    tags, W, D = ctx["tags"], ctx["W"], ctx["D"]
    T, DR, KF, fix = log["true"], log["dr"], log["kf"], log["fix"]
    e_dr = np.hypot(*(DR - T)[:, :2].T)
    e_kf = np.hypot(*(KF - T)[:, :2].T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), W, D, fill=False, ec=AXIS, lw=1.5))
    for t in tags:
        ax.plot(*t.center[:2], "s", color=INK2, ms=4)
    ax.plot(T[:, 0], T[:, 1], color=INK, lw=2, label="True path")
    ax.plot(DR[:, 0], DR[:, 1], color=C["multi"], lw=2,
            label="Motor dead reckoning")
    ax.plot(KF[:, 0], KF[:, 1], color=C["single"], lw=2, ls="--",
            label="Dead reckoning + tag fixes")
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, W + 0.6)
    ax.set_ylim(-0.6, D + 0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Where the robot thinks it is (squares = tags)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)

    ax = axes[1]
    ax.plot(e_dr * 100, color=C["multi"], lw=2, label="Motor dead reckoning")
    ax.plot(e_kf * 100, color=C["single"], lw=2, label="Dead reckoning + tag fixes")
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xlabel("Step")
    ax.set_ylabel("Position error (cm)")
    ax.set_title("Position error over time", loc="left")
    ax.text(len(e_dr) - 1, e_dr[-1] * 100, f" {e_dr[-1] * 100:.0f} cm",
            color=INK2, va="center", fontsize=9)
    ax.text(len(e_kf) - 1, max(e_kf[-1] * 100, 0.2), f" {e_kf[-1] * 100:.1f} cm",
            color=INK2, va="center", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "trajectory.png"), dpi=150)
    plt.close(fig)
    return {"steps": steps, "fix_rate_pct": 100 * fix.mean(),
            "dr_final_err_cm": 100 * e_dr[-1], "dr_max_err_cm": 100 * e_dr.max(),
            "fused_median_err_cm": 100 * np.median(e_kf),
            "fused_p90_err_cm": 100 * np.percentile(e_kf, 90),
            "fused_max_err_cm": 100 * e_kf.max()}


if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else "results/localization"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    example_frame()
    print("A. Monte Carlo ...", flush=True)
    A = experiment_A()
    print(json.dumps(A, indent=1, default=float), flush=True)
    print("B. Sensitivity ...", flush=True)
    B = experiment_B()
    print("C. Trajectory ...", flush=True)
    Cr = experiment_C()
    print(json.dumps(Cr, indent=1, default=float))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"A_main": A, "B_sensitivity": B, "C_trajectory": Cr}, f,
                  indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}/")
