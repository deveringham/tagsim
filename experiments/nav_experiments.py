"""
nav_experiments.py - how few tags does the robot need to get around?

  python3 -m experiments.nav_experiments [output_dir] [--workers N] [--only K 1 2 2b 3]
  (from the repo root; default output: results/navigation)

The robot's motors have unknown, slowly drifting biases (nav.Motors):
step-length scale s, turn scale k and curve per step c.  The pose filter
either ignores them (large process noise) or learns them online
(nav.OdometryEKF(calibrate=True)).

K. Calibration on the existing experiments: experiment C's loop (all 18
   tags) and experiment E's cube runs, with online calibration off / on,
   with and without bias drift.
1. Fix quality map: the robot stands at every point of a grid facing a
   single landmark (one wall tag, a 10 cm cube, a 15 cm cube) and takes
   photos.  Where is it detected, and how good is the pose fix?
2. Blind walking: after a warm-up with tag fixes, the robot turns away
   and walks with no fixes.  Pose error vs distance walked for a filter
   that ignores the motor biases, one that learned them during the
   warm-up, and one told the true biases.  The warm-up fixes are
   synthetic (true pose + Gaussian noise) and the blind leg is a pure
   motion model, so no images are rendered.
2b. As 2, but the warm-up is a real run in the arena: rendered photos,
   detected tags and the Navigator (18 wall tags, or only the 10 cm cube),
   so fixes have real outliers, correlated errors and gaps.
3. Reaching spots: the robot is sent to a series of random spots using
   only 1 wall tag, 1 cube, 2, 4 or all 18 wall tags, turning to look at a
   landmark whenever it gets unsure of its position (nav.Navigator),
   with calibration off / on.  True error on arrival, steps taken, looks.
"""
import sys, os, json, time, argparse, copy
from multiprocessing import get_context
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from tagsim import *
from nav import *
from experiments.run_experiments import (realistic_setup, simulate_run, plain_log, INK,
                             INK2, MUTED, GRID, AXIS, SURF, C)
import experiments.cube_experiments as CX

OUT = "results/navigation"  # set from the command line in __main__
WORKERS = 8

STAND = Degradation(bob_pitch_deg=0.7, bob_roll_deg=0.5, bob_dz_m=0.002)
OFF_ON = {False: ("Biases ignored", C["multi"]),
          True: ("Biases learned online", C["single"])}
SEQ = LinearSegmentedColormap.from_list("seq", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ.set_bad("#efeee9")
DETECT_PX = 18.0          # tag side (px) beyond which detection gets unreliable
CUBE_CENTRE = np.array([2.0, 1.5])
SETUPS = {                # landmark sets; wall-tag ids are make_arena()'s
    "1 wall tag": {"ids": [2]},                   # middle of the south wall
    "1 cube (10 cm)": {"cube": (0.10, 0.07)},     # in the middle of the room
    "1 cube (15 cm)": {"cube": (0.15, 0.11)},
    "2 wall tags": {"ids": [2, 11]},              # middles of opposite walls
    "4 wall tags": {"ids": [2, 7, 11, 16]},       # one per wall
    "18 wall tags": {"ids": None},                # the full arena
}


def pool():
    return get_context("fork").Pool(WORKERS)


def make_scene(setup, rng):
    """True tags, the robot's map of them, camera and landmarks for a setup."""
    tags, map_tags, cam, K_est, (W, D) = realistic_setup(rng)
    spec = SETUPS[setup]
    cube = None
    if "cube" in spec:
        edge, ts = spec["cube"]
        yaw = np.radians(20)
        cube = Cube(make_T(rot_z(yaw), [*CUBE_CENTRE, edge / 2]), edge, ts)
        # Tape-measure error on where the cube is: 1 cm, 1 deg (as walls).
        dp = rng.normal(0, 0.01, 2)
        T_map = make_T(rot_z(yaw + np.radians(rng.normal(0, 1.0))),
                       [*(CUBE_CENTRE + dp), edge / 2])
        true, mapped = cube.tags(), cube.tags(T_map)
        lms = [Landmark(T_map[:2, 3], None, ts * cam.f / DETECT_PX)]
    else:
        keep = lambda t: spec["ids"] is None or t.id in spec["ids"]
        true = [t for t in tags if keep(t)]
        mapped = [t for t in map_tags if keep(t)]
        lms = [Landmark(t.center[:2], t.normal[:2], t.size * cam.f / DETECT_PX)
               for t in mapped]
    return {"tags": true, "map_tags": mapped, "cam": cam, "K_est": K_est,
            "W": W, "D": D, "landmarks": lms, "cube": cube}


def draw_landmarks(ax, sc, **kw):
    if sc["cube"] is not None:
        c = sc["cube"].corners()[:, :2]
        ax.fill(*c[[0, 2, 6, 4]].T, color=INK, zorder=5, **kw)
    for t in ([] if sc["cube"] is not None else sc["tags"]):
        ax.plot(*t.center[:2], "s", color=INK, ms=6, zorder=5, **kw)


# ---------------------------------------------------------------------------
# K. Calibration on the existing experiments
# ---------------------------------------------------------------------------
def _c_run(args):
    drift, cal = args
    log, _ = simulate_run(240, motors=Motors(drift=drift), calibrate=cal)
    return log


def _e_run(args):
    seed, drift, cal = args
    log, ctx = CX.cube_run(seed, 320, motors=Motors(drift=drift), calibrate=cal)
    return CX.run_stats(log, ctx)


def experiment_K():
    drifts = {"no drift": (0.0, 0.0, 0.0), "drift": DRIFT}
    cases = [(d, cal) for d in drifts for cal in (False, True)]
    with pool() as p:
        c_logs = p.map(_c_run, [(drifts[d], cal) for d, cal in cases])
        e_stats = p.map(_e_run, [(s, drifts[d], cal) for d, cal in cases
                                 for s in range(8)])
    out = {"C_loop_18_tags": {}, "E_cube_runs": {}}
    for i, (d, cal) in enumerate(cases):
        name = f"{d}, calibration {'on' if cal else 'off'}"
        log = c_logs[i]
        e = 100 * np.hypot(*(log["kf"] - log["true"])[:, :2].T)
        bt, be = log["bias_true"][-1], log["bias_est"][-1]
        out["C_loop_18_tags"][name] = {
            "pos_median_cm": np.median(e), "pos_p90_cm": np.percentile(e, 90),
            "pos_max_cm": e.max(), "fix_accept_pct": 100 * log["fix"].mean(),
            "final_true_s_k_cdeg": [bt[0], bt[1], np.degrees(bt[2])],
            "final_est_s_k_cdeg": [be[0], be[1], np.degrees(be[2])]}
        st = e_stats[8 * i: 8 * i + 8]
        out["E_cube_runs"][name] = {
            k: float(np.median([s[k] for s in st])) for k in (
                "per_photo_map_err_median_cm", "per_photo_map_err_p90_cm",
                "placement_spread_rms_cm", "fused_final_err_cm",
                "fused_final_rot_deg")}
        out["E_cube_runs"][name]["note"] = "median over 8 runs"
    plot_K(c_logs[2], c_logs[3])
    return out


def plot_K(log_off, log_on):
    """Experiment C with drifting biases: what the filter learns."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    k = np.arange(1, len(log_on["true"]) + 1)
    names = ["Step length scale s", "Turn scale k", "Curve c (deg / step)"]
    for i, ax in enumerate(axes[:3]):
        f = np.degrees if i == 2 else (lambda v: v)
        est, sd = f(log_on["bias_est"][:, i]), f(log_on["bias_sd"][:, i])
        ax.fill_between(k, est - 2 * sd, est + 2 * sd, color=C["single"],
                        alpha=0.18, lw=0)
        ax.plot(k, est, color=C["single"], lw=2, label="Learned (± 2 sd)")
        ax.plot(k, f(log_on["bias_true"][:, i]), color=INK, lw=1.5, ls="--",
                label="True (drifting)")
        ax.set_title(names[i], loc="left")
        ax.set_xlabel("Step")
    lo = [0.6, 0.7, -3]
    hi = [1.2, 1.2, 4]
    for ax, a, b in zip(axes[:3], lo, hi):
        ax.set_ylim(a, b)
    axes[0].legend(loc="lower right", fontsize=8.5)
    ax = axes[3]
    for cal, log in ((False, log_off), (True, log_on)):
        e = 100 * np.hypot(*(log["kf"] - log["true"])[:, :2].T)
        ax.plot(k, e, color=OFF_ON[cal][1], lw=1.5, label=OFF_ON[cal][0])
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_ylim(0.1, 40)
    ax.set_title("Position error (cm)", loc="left")
    ax.set_xlabel("Step")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    fig.suptitle("Experiment C loop (fix in almost every photo) with drifting "
                 "motor biases", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "calibration_loop.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Fix quality map
# ---------------------------------------------------------------------------
GRID_X = np.arange(0.25, 3.76, 0.25)
GRID_Y = np.arange(0.25, 2.76, 0.25)


def _fix_map(args):
    setup, photos = args
    rng = np.random.default_rng(77)
    sc = make_scene(setup, rng)
    det = make_detector()
    lm = sc["landmarks"][0]
    shape = (len(GRID_Y), len(GRID_X))
    out = {k: np.full(shape, np.nan) for k in ("detect", "median", "gross", "dist")}
    for iy, y in enumerate(GRID_Y):
        for ix, x in enumerate(GRID_X):
            p = np.array([x, y])
            if sc["cube"] is not None and np.linalg.norm(p - CUBE_CENTRE) < 0.35:
                continue
            face = np.arctan2(*(lm.xy - p)[::-1])
            hit, err = 0, []
            for _ in range(photos):
                pose = (x, y, face + np.radians(rng.uniform(-10, 10)))
                r = measure(pose, sc["tags"], sc["cam"], STAND, rng, detector=det,
                            K_est=sc["K_est"], map_tags=sc["map_tags"])
                if r.n_tags:
                    hit += 1
                    err.append(r.err["floor"][0])
            out["detect"][iy, ix] = 100 * hit / photos
            out["dist"][iy, ix] = np.linalg.norm(p - lm.xy)
            if err:
                out["median"][iy, ix] = 100 * np.median(err)
                out["gross"][iy, ix] = 100 * np.mean(np.array(err) > 0.25)
    return out


def experiment_1(photos=8):
    setups = ["1 wall tag", "1 cube (10 cm)", "1 cube (15 cm)"]
    with pool() as p:
        maps = dict(zip(setups, p.map(_fix_map, [(s, photos) for s in setups])))
    plot_1(maps)
    summary = {}
    for s, m in maps.items():
        valid = ~np.isnan(m["detect"])
        n = valid.sum()
        ok = lambda cm: 100 * np.sum(valid & (m["detect"] >= 90) &
                                     (np.nan_to_num(m["median"], nan=1e9) < cm)) / n
        far = m["dist"][valid & (m["detect"] >= 90)]
        summary[s] = {
            "grid_points": int(n),
            "area_detect_ge90_pct": 100 * np.sum(valid & (m["detect"] >= 90)) / n,
            "area_detect_ge90_and_median_lt5cm_pct": ok(5),
            "area_detect_ge90_and_median_lt10cm_pct": ok(10),
            "max_dist_detect_ge90_m": float(far.max()) if len(far) else np.nan,
            "gross_fail_pct_of_fixes (>25 cm)": float(np.nansum(
                m["gross"] * m["detect"]) / np.nansum(m["detect"])),
            "median_err_cm_by_dist": {
                f"{a}-{b} m": float(np.nanmedian(m["median"][(m["dist"] >= a) &
                                                             (m["dist"] < b)]))
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 4.5))
                if np.any(~np.isnan(m["median"][(m["dist"] >= a) & (m["dist"] < b)]))}}
    return summary


def plot_1(maps):
    rows = [("detect", "Photos with a fix (%)", None, (0, 100)),
            ("median", "Median position error (cm)", "log", (0.5, 50)),
            ("gross", "Fixes off by > 25 cm (%)", None, (0, 40))]
    fig, axes = plt.subplots(3, len(maps), figsize=(4.6 * len(maps) + 1, 9.6))
    ext = [GRID_X[0] - 0.125, GRID_X[-1] + 0.125, GRID_Y[0] - 0.125, GRID_Y[-1] + 0.125]
    for j, (setup, m) in enumerate(maps.items()):
        sc = make_scene(setup, np.random.default_rng(77))
        for i, (key, title, scale, (lo, hi)) in enumerate(rows):
            ax = axes[i, j]
            norm = LogNorm(lo, hi) if scale else plt.Normalize(lo, hi)
            im = ax.imshow(m[key], origin="lower", extent=ext, cmap=SEQ, norm=norm,
                           interpolation="nearest")
            ax.add_patch(Rectangle((0, 0), sc["W"], sc["D"], fill=False, ec=AXIS,
                                   lw=1.5))
            draw_landmarks(ax, sc)
            ax.set_xlim(-0.1, sc["W"] + 0.1)
            ax.set_ylim(-0.1, sc["D"] + 0.1)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(setup, loc="left")
            if j == len(maps) - 1:
                cb = fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.9, pad=0.02)
                cb.set_label(title, color=INK2)
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=8, colors=MUTED)
                if scale:
                    cb.ax.yaxis.set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    fig.suptitle("Fix quality from a single landmark: robot standing at each "
                 "point, facing it (grey = no fix)", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fix_quality_map.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Blind walking
# ---------------------------------------------------------------------------
LEG_STEPS = 30                      # 3 m of commanded walking
MODES = {"ignored": "Biases ignored", "learned": "Biases learned online",
         "oracle": "True biases known"}
MODE_C = {"ignored": C["multi"], "learned": C["single"], "oracle": C["floor"]}


def blind_trial(seed, warmup):
    """One warm-up (fix after every step) then one blind turn-and-walk.
    Returns pos / heading error per walking step for each mode, and the
    learned-bias errors."""
    rng = np.random.default_rng(seed)
    drift_rng = np.random.default_rng(seed + 10 ** 6)
    motors = random_motors(rng)
    x_true = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)])
    ekf = OdometryEKF(x_true, calibrate=True)
    for _ in range(warmup):
        if rng.random() < 0.6:
            fwd, turn = 0.10, np.radians(rng.uniform(-10, 10))
        else:
            fwd, turn = 0.0, np.radians(rng.choice([-1, 1]) * rng.uniform(10, 30))
        x_true = motors.move(x_true, fwd, turn, rng, drift_rng)
        ekf.predict(fwd, turn)
        R = fix_cov(rng.uniform(0.5, 2.5))
        z = x_true + rng.multivariate_normal(np.zeros(3), R)
        ekf.update(z, R)
    return blind_leg(motors, x_true, ekf, rng, drift_rng)


def blind_leg(motors, x_true, ekf, rng, drift_rng):
    """From the filter's current state: turn away 90-180 deg, then walk
    LEG_STEPS with no fixes, predicting with ignored / learned / true
    biases.  Returns pos and heading error per walking step for each mode,
    and the learned-bias errors."""
    bias_err = np.abs(ekf.biases - motors.biases)
    models = {"ignored": np.array([1.0, 1.0, 0.0]), "learned": ekf.biases.copy(),
              "oracle": motors.biases.copy()}
    est = {m: ekf.pose.copy() for m in models}
    # Turn away (90-180 deg, at most 30 deg per command), then walk straight.
    total = rng.choice([-1, 1]) * np.radians(rng.uniform(90, 180))
    n_turn = int(np.ceil(abs(total) / np.radians(30)))
    cmds = [(0.0, total / n_turn)] * n_turn + [(0.10, 0.0)] * LEG_STEPS
    pos = {m: [] for m in models}
    yaw = {m: [] for m in models}
    for i, (fwd, turn) in enumerate(cmds):
        x_true = motors.move(x_true, fwd, turn, rng, drift_rng)
        for m, (s, k, c) in models.items():
            est[m] = motion(est[m], s * fwd, k * turn + c * (fwd > 0))
            if i >= n_turn:
                pos[m].append(np.hypot(*(est[m][:2] - x_true[:2])))
                yaw[m].append(abs(np.degrees(wrap(est[m][2] - x_true[2]))))
    return pos, yaw, bias_err


def _blind_batch(args):
    warmup, seeds = args
    return [blind_trial(s, warmup) for s in seeds]


def experiment_2(trials=800, warmups=(0, 10, 25, 50, 100, 200), main_warmup=100):
    jobs = [(w, range(i, trials, WORKERS)) for w in warmups for i in range(WORKERS)]
    with pool() as p:
        res = p.map(_blind_batch, [(w, list(s)) for w, s in jobs])
    by_w = {w: [] for w in warmups}
    for (w, _), r in zip(jobs, res):
        by_w[w] += r
    dist = 0.10 * np.arange(1, LEG_STEPS + 1)

    def curves(trs, m, kind):
        a = np.array([t[0 if kind == "pos" else 1][m] for t in trs])
        return np.median(a, 0), np.percentile(a, 90, axis=0)

    def reach(curve, limit):
        """Commanded distance walked before the curve passes the limit."""
        over = np.nonzero(curve > limit)[0]
        return float(dist[over[0] - 1]) if len(over) and over[0] > 0 else (
            0.0 if len(over) else float(dist[-1]))

    out = {"main_warmup_steps": main_warmup, "trials_per_warmup": trials,
           "modes": {}, "learned_vs_warmup": {}}
    trs = by_w[main_warmup]
    for m in MODES:
        pm, p90 = curves(trs, m, "pos")
        ym, y90 = curves(trs, m, "yaw")
        out["modes"][MODES[m]] = {
            "blind_m_until_median_err_10cm": reach(pm, 0.10),
            "blind_m_until_p90_err_10cm": reach(p90, 0.10),
            "blind_m_until_median_err_20cm": reach(pm, 0.20),
            "blind_m_until_p90_heading_err_30deg": reach(y90, 30),
            "err_at_1m_median_cm": 100 * pm[9], "err_at_1m_p90_cm": 100 * p90[9],
            "err_at_2m_median_cm": 100 * pm[19], "err_at_2m_p90_cm": 100 * p90[19],
            "heading_err_at_1m_median_deg": ym[9]}
    for w in warmups:
        pm, p90 = curves(by_w[w], "learned", "pos")
        be = np.array([t[2] for t in by_w[w]])
        out["learned_vs_warmup"][w] = {
            "blind_m_until_median_err_10cm": reach(pm, 0.10),
            "blind_m_until_p90_err_10cm": reach(p90, 0.10),
            "s_err_median_pct": 100 * np.median(be[:, 0]),
            "k_err_median_pct": 100 * np.median(be[:, 1]),
            "c_err_median_deg": float(np.degrees(np.median(be[:, 2])))}
    plot_2(by_w, main_warmup, dist, curves, out)
    return out


def plot_2(by_w, main_warmup, dist, curves, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    trs = by_w[main_warmup]
    for ax, kind, unit, title, line in (
            (axes[0], "pos", 100, "Position error (cm)", 10),
            (axes[1], "yaw", 1, "Heading error (deg)", 30)):
        for m in MODES:
            med, p90 = curves(trs, m, kind)
            ax.plot(dist, med * unit, color=MODE_C[m], lw=2, label=MODES[m])
            ax.plot(dist, p90 * unit, color=MODE_C[m], lw=1.2, ls="--")
        ax.axhline(line, color=MUTED, lw=1, ls=":")
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlabel("Distance walked blind (m, commanded)")
        ax.set_title(f"{title}: median (solid), 90th pct (dashed)", loc="left",
                     fontsize=10)
    axes[0].text(dist[-1], 10, " 10 cm", color=MUTED, va="bottom", ha="right",
                 fontsize=8.5)
    axes[1].text(dist[-1], 30, " 30 deg", color=MUTED, va="bottom", ha="right",
                 fontsize=8.5)
    ax = axes[2]
    ws = list(out["learned_vs_warmup"])
    for key, ls, lab in (("blind_m_until_median_err_10cm", "-", "median"),
                         ("blind_m_until_p90_err_10cm", "--", "90th pct")):
        ax.plot(ws, [out["learned_vs_warmup"][w][key] for w in ws], color=C["single"],
                lw=2, ls=ls, marker="o", ms=4, label=f"Learned, {lab}")
        ax.axhline(out["modes"][MODES["oracle"]][key], color=C["floor"], lw=1.2,
                   ls=ls, label=f"True biases known, {lab}")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(ws)
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Warm-up steps with tag fixes (0 = biases ignored)")
    ax.set_title("Blind distance before error > 10 cm (m)", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left",
               bbox_to_anchor=(0.01, 0.0), ncol=3, fontsize=9)
    fig.suptitle(f"Walking blind after a turn (random robots, drifting biases, "
                 f"{main_warmup}-step warm-up)", x=0.01, ha="left", color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "blind_walk.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2b. Blind walking after a rendered warm-up
# ---------------------------------------------------------------------------
WARMUPS_2B = (10, 25, 50, 100, 200)


def blind_trial_rendered(args):
    """Warm up with real photos while walking to random spots (looking at
    landmarks when unsure).  At each warm-up length, right after the next
    accepted fix, branch off a blind leg from a copy of the state."""
    setup, seed = args
    rng = np.random.default_rng(seed)
    sc = make_scene(setup, np.random.default_rng(500 + seed))
    motors = random_motors(np.random.default_rng(600 + seed))
    drift_rng = np.random.default_rng(700 + seed)
    goals = iter(goal_sequence(np.random.default_rng(800 + seed), 400,
                               sc["W"], sc["D"]))
    det = make_detector()
    ekf = OdometryEKF(START, calibrate=True)
    nav = Navigator(sc["landmarks"])
    nav.set_goal(next(goals))
    x_true, fixed, n_fix = START.copy(), False, 0
    out, todo = {}, list(WARMUPS_2B)
    step = 0
    while todo and step < 2 * max(WARMUPS_2B):
        cmd = nav.command(ekf.pose, ekf.pos_sd, fixed)
        while cmd is None:
            nav.set_goal(next(goals))
            cmd = nav.command(ekf.pose, ekf.pos_sd, fixed)
        fwd, turn = cmd
        x_true = motors.move(x_true, fwd, turn, rng, drift_rng)
        x_true[:2] = np.clip(x_true[:2], 0.25, [sc["W"] - 0.25, sc["D"] - 0.25])
        ekf.predict(fwd, turn)
        r = measure(tuple(x_true), sc["tags"], sc["cam"], STAND, rng, detector=det,
                    K_est=sc["K_est"], map_tags=sc["map_tags"])
        fixed = r.n_tags > 0 and ekf.update(np.array(r.est["floor"]),
                                            fix_cov(r.nearest_dist))
        step += 1
        n_fix += fixed
        if fixed and step >= todo[0]:
            w = todo.pop(0)
            leg_rng = np.random.default_rng([seed, w])
            out[w] = blind_leg(Motors(**vars(motors)), x_true.copy(),
                               copy.deepcopy(ekf), leg_rng, leg_rng) + (step,)
    return {"legs": out, "fix_pct": 100 * n_fix / max(step, 1)}


def experiment_2b(trials=300, setups=("18 wall tags", "1 cube (10 cm)")):
    with pool() as p:
        res = p.map(blind_trial_rendered, [(s, seed) for s in setups
                                           for seed in range(trials)])
    syn = json.load(open(os.path.join(OUT, "nav_results.json")))[
        "2_blind_walking"]["learned_vs_warmup"]
    out = {"trials_per_setup": trials, "synthetic_fixes (experiment 2)": {
        w: syn[str(w)] for w in WARMUPS_2B if str(w) in syn}}
    reach = lambda c: (float(0.1 * np.nonzero(c > 0.10)[0][0])
                       if np.any(c > 0.10) else 0.1 * LEG_STEPS)
    for i, s in enumerate(setups):
        rs = res[i * trials:(i + 1) * trials]
        out[f"rendered, {s}"] = d = {"warmup_fix_pct_mean": float(np.mean(
            [r["fix_pct"] for r in rs]))}
        for w in WARMUPS_2B:
            legs = [r["legs"][w] for r in rs if w in r["legs"]]
            pos = {m: np.array([l[0][m] for l in legs]) for m in MODES}
            be = np.array([l[2] for l in legs])
            d[w] = {"trials": len(legs),
                    "actual_warmup_steps_median": float(np.median([l[3] for l in legs])),
                    "blind_m_until_median_err_10cm": reach(np.median(pos["learned"], 0)),
                    "blind_m_until_p90_err_10cm": reach(np.percentile(pos["learned"], 90, 0)),
                    "oracle_blind_m_until_median_err_10cm": reach(np.median(pos["oracle"], 0)),
                    "s_err_median_pct": 100 * np.median(be[:, 0]),
                    "k_err_median_pct": 100 * np.median(be[:, 1]),
                    "c_err_median_deg": float(np.degrees(np.median(be[:, 2])))}
    plot_2b(out, setups)
    return out


def plot_2b(out, setups):
    srcs = [("synthetic_fixes (experiment 2)", "Synthetic fixes (experiment 2)", INK2)]
    srcs += [(f"rendered, {s}", f"Rendered photos, {s}", c)
             for s, c in zip(setups, (C["single"], C["multi"]))]
    panels = [("blind_m_until_median_err_10cm", "Blind distance before median error > 10 cm (m)"),
              ("s_err_median_pct", "Step-scale error s (%, median)"),
              ("k_err_median_pct", "Turn-scale error k (%, median)"),
              ("c_err_median_deg", "Curve error c (deg / step, median)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    ws = list(WARMUPS_2B)
    for ax, (key, title) in zip(axes, panels):
        for src, lab, col in srcs:
            ax.plot(ws, [out[src][w][key] for w in ws], color=col, lw=2,
                    marker="o", ms=4, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(ws)
        ax.set_xticklabels([str(w) for w in ws])
        ax.minorticks_off()
        ax.set_ylim(0, None)
        ax.set_xlabel("Warm-up steps")
        ax.set_title(title, loc="left", fontsize=10)
    # Same runs with the true biases: the ceiling for each warm-up source.
    for src, lab, col in srcs[1:]:
        axes[0].plot(ws, [out[src][w]["oracle_blind_m_until_median_err_10cm"]
                          for w in ws], color=col, lw=1.2, ls="--")
    axes[0].text(0.02, 0.03, "dashed: same runs with the true biases",
                 transform=axes[0].transAxes, color=INK2, fontsize=8.5)
    axes[0].set_ylim(0, 1.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left",
               bbox_to_anchor=(0.01, 0.0), ncol=3, fontsize=9)
    fig.suptitle("Learning the motor biases from real (rendered) tag fixes vs "
                 "idealised ones", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "blind_walk_rendered.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Reaching spots
# ---------------------------------------------------------------------------
START = np.array([0.7, 0.7, 0.0])


def goal_sequence(rng, n, W, D, margin=0.4, clear=0.35):
    """Random spots; neither they nor the straight lines between them pass
    within `clear` of the room centre (where the cube sits)."""
    def seg_dist(a, b, p):
        t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        return np.linalg.norm(a + t * (b - a) - p)
    goals, p = [], START[:2]
    while len(goals) < n:
        g = rng.uniform([margin, margin], [W - margin, D - margin])
        if (np.linalg.norm(g - p) > 0.5 and np.linalg.norm(g - CUBE_CENTRE) > clear
                and seg_dist(p, g, CUBE_CENTRE) > clear):
            goals.append(g)
            p = g
    return goals


def goal_run(args):
    setup, cal, seed, n_goals, max_steps = args
    rng = np.random.default_rng(seed)             # images, slip
    sc = make_scene(setup, np.random.default_rng(500 + seed))
    motors = random_motors(np.random.default_rng(600 + seed))
    drift_rng = np.random.default_rng(700 + seed)
    goals = goal_sequence(np.random.default_rng(800 + seed), n_goals,
                          sc["W"], sc["D"])
    det = make_detector()
    ekf = OdometryEKF(START, calibrate=cal)
    nav = Navigator(sc["landmarks"])
    x_true, fixed = START.copy(), False
    records, path = [], [x_true.copy()]
    for g in goals:
        nav.set_goal(g)
        s0 = dict(nav.stats)
        steps = 0
        while True:
            cmd = nav.command(ekf.pose, ekf.pos_sd, fixed)
            if cmd is None or steps >= max_steps:
                break
            fwd, turn = cmd
            x_true = motors.move(x_true, fwd, turn, rng, drift_rng)
            x_true[:2] = np.clip(x_true[:2], 0.25, [sc["W"] - 0.25, sc["D"] - 0.25])
            ekf.predict(fwd, turn)
            r = measure(tuple(x_true), sc["tags"], sc["cam"], STAND, rng,
                        detector=det, K_est=sc["K_est"], map_tags=sc["map_tags"])
            fixed = r.n_tags > 0 and ekf.update(np.array(r.est["floor"]),
                                                fix_cov(r.nearest_dist))
            steps += 1
            path.append(x_true.copy())
        records.append({
            "err_m": float(np.linalg.norm(x_true[:2] - g)),
            "arrived": cmd is None, "steps": steps,
            "confirmed": bool(ekf.pos_sd <= nav.look_sd),
            "looks": nav.stats["looks"] - s0["looks"],
            "searches": nav.stats["searches"] - s0["searches"],
            "gave_up": nav.stats["gave_up"] - s0["gave_up"],
            "dist_m": float(np.linalg.norm(g - path[-steps - 1][:2])) if steps else 0.0})
    return {"setup": setup, "cal": cal, "seed": seed, "goals": records,
            "path": np.array(path), "goal_xy": np.array(goals),
            "bias_true": motors.biases, "bias_est": ekf.biases}


def experiment_3(seeds=8, n_goals=12, max_steps=200):
    setups = ["1 wall tag", "1 cube (10 cm)", "2 wall tags", "4 wall tags",
              "18 wall tags"]
    jobs = [(s, cal, seed, n_goals, max_steps) for s in setups
            for cal in (False, True) for seed in range(seeds)]
    with pool() as p:
        runs = p.map(goal_run, jobs)
    out = {}
    for s in setups:
        for cal in (False, True):
            g = [r for run in runs if run["setup"] == s and run["cal"] == cal
                 for r in run["goals"]]
            e = 100 * np.array([r["err_m"] for r in g])
            out.setdefault(s, {})["on" if cal else "off"] = {
                "goals": len(g),
                "err_median_cm": np.median(e), "err_p90_cm": np.percentile(e, 90),
                "within_10cm_pct": 100 * np.mean(e <= 10),
                "within_20cm_pct": 100 * np.mean(e <= 20),
                "timeouts_pct": 100 * np.mean([not r["arrived"] for r in g]),
                "arrived_without_fresh_fix_pct": 100 * np.mean(
                    [r["arrived"] and not r["confirmed"] for r in g]),
                "steps_per_goal_median": float(np.median([r["steps"] for r in g])),
                "steps_per_m_median": float(np.median(
                    [r["steps"] / r["dist_m"] for r in g if r["dist_m"] > 0.3])),
                "looks_per_goal_mean": float(np.mean([r["looks"] for r in g])),
                "searches_per_goal_mean": float(np.mean([r["searches"] for r in g]))}
    plot_3(out, setups)
    plot_3_paths(runs)
    return out


def plot_3(out, setups):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(setups))
    for cal, dx in ((False, -0.12), (True, 0.12)):
        key = "on" if cal else "off"
        lab, col = OFF_ON[cal]
        med = [out[s][key]["err_median_cm"] for s in setups]
        p90 = [out[s][key]["err_p90_cm"] for s in setups]
        ax = axes[0]
        ax.vlines(x + dx, med, p90, color=col, lw=2)
        ax.plot(x + dx, med, "o", color=col, ms=7, label=lab)
        ax.plot(x + dx, p90, "_", color=col, ms=12, mew=2)
        axes[1].plot(x + dx, [out[s][key]["within_10cm_pct"] for s in setups], "o",
                     color=col, ms=7)
        axes[1].plot(x + dx, [out[s][key]["within_20cm_pct"] for s in setups], "^",
                     color=col, ms=7, mfc="none", mew=1.5)
        axes[2].bar(x + dx, [out[s][key]["steps_per_m_median"] for s in setups],
                    0.22, color=col, edgecolor=SURF, lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1, 30)
    plain_log(axes[0])
    axes[0].set_title("True error on arrival (cm): median dot, 90th pct tick",
                      loc="left", fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Spots reached within 10 cm (dot) / 20 cm (triangle), %",
                      loc="left", fontsize=10)
    axes[2].set_title("Steps per metre to the spot (median)", loc="left",
                      fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" (", "\n(") for s in setups], fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left",
               bbox_to_anchor=(0.01, 0.0), ncol=2, fontsize=9)
    fig.suptitle("Reaching random spots with few landmarks (random drifting "
                 "robots, turning to look when unsure)", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "goal_reaching.png"), dpi=150)
    plt.close(fig)


def plot_3_paths(runs, seed=0, setups=("1 wall tag", "1 cube (10 cm)")):
    """True paths for one robot, biases ignored vs learned."""
    fig, axes = plt.subplots(len(setups), 2, figsize=(11, 4.0 * len(setups)))
    for i, s in enumerate(setups):
        sc = make_scene(s, np.random.default_rng(500 + seed))
        for j, cal in enumerate((False, True)):
            run = next(r for r in runs if r["setup"] == s and r["cal"] == cal
                       and r["seed"] == seed)
            ax = axes[i, j]
            ax.add_patch(Rectangle((0, 0), sc["W"], sc["D"], fill=False, ec=AXIS,
                                   lw=1.5))
            draw_landmarks(ax, sc)
            ax.plot(*run["path"][:, :2].T, color=OFF_ON[cal][1], lw=1, alpha=0.8)
            gx = run["goal_xy"]
            ax.plot(gx[:, 0], gx[:, 1], "x", color=INK, ms=7, mew=2)
            for gxy, rec in zip(gx, run["goals"]):
                ax.annotate(f"{100 * rec['err_m']:.0f}", gxy, xytext=(4, 4),
                            textcoords="offset points", fontsize=8, color=INK2)
            e = np.array([r["err_m"] for r in run["goals"]]) * 100
            ax.set_title(f"{s}, {OFF_ON[cal][0].lower()}: median {np.median(e):.0f} cm",
                         loc="left", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-0.1, sc["W"] + 0.1)
            ax.set_ylim(-0.1, sc["D"] + 0.1)
    fig.suptitle("One robot's true paths (x = spots, numbers = arrival error in cm)",
                 x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "goal_paths.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/navigation")
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    ap.add_argument("--only", nargs="*", default=["K", "1", "2", "2b", "3"])
    a = ap.parse_args()
    OUT, WORKERS = a.out, a.workers
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "nav_results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for key, name, fn in (("K", "K_calibration_on_existing", experiment_K),
                          ("1", "1_fix_quality_map", experiment_1),
                          ("2", "2_blind_walking", experiment_2),
                          ("2b", "2b_blind_walking_rendered_warmup", experiment_2b),
                          ("3", "3_reaching_spots", experiment_3)):
        if key in a.only:
            print(f"{name} ...", flush=True)
            results[name] = fn()
            print(json.dumps(results[name], indent=1, default=float), flush=True)
            with open(path, "w") as f:
                json.dump(results, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}/")
