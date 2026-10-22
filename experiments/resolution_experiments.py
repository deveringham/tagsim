"""
resolution_experiments.py - what does camera resolution buy?

  python3 -m experiments.resolution_experiments [output_dir] [--workers N] [--only R1 R2]
  (from the repo root; default output: results/resolution)

Same 70 deg lens and 4:3 aspect throughout; resolutions 320x240 to
1920x1440.  Blur, noise and JPEG are per pixel as before, i.e. the lens is
assumed sharp enough for the sensor - except in the "lens-limited" runs,
where the blur is fixed in angle (0.7 px at 640 wide, scaled up with the
resolution), so extra pixels add no detail.  The detector's minimum marker
size is held at the same pixel size at every resolution (make_detector).

R1. Fix quality vs distance at each resolution, for the 10 cm cube and a
    15 cm wall tag (as scale_experiments S1).  Reach of detection and of
    10 cm fixes vs resolution; the cube size at 640 x 480 that would reach
    as far; detection time per frame.
R2. Reaching spots with one 10 cm cube in the middle of 4-12 m rooms at
    each resolution (as scale_experiments S2): where does the cusp move?
"""
import sys, os, json, time, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tagsim import *
import experiments.nav_experiments as N
import experiments.scale_experiments as S
from experiments.run_experiments import plain_log, INK, INK2, MUTED, SURF, C

OUT = "results/resolution"            # set from the command line in __main__

RES = [(320, 240), (640, 480), (960, 720), (1280, 960), (1920, 1440)]
LANDMARKS = [("cube", 0.10), ("wall", 0.15)]
ROOMS_W = (4.0, 6.0, 8.0, 12.0)
BLUR_PX = 0.7                         # default blur at 640 wide (Degradation)


def res_label(wh):
    return f"{wh[0]} x {wh[1]}"


# ---------------------------------------------------------------------------
# R1. Fix quality vs distance at each resolution
# ---------------------------------------------------------------------------
def detect_time_ms(wh, reps=20):
    """Detection time for one frame of the arena (this machine, 1 core)."""
    cam = Camera(width=wh[0], height=wh[1])
    tags, _ = make_arena()
    rng = np.random.default_rng(0)
    img = render(tags, cam, robot_T(2.0, 1.0, 1.2) @ cam.T_robot_cam(),
                 N.STAND, rng)
    det = make_detector(cam)
    det.detectMarkers(img)
    t = time.perf_counter()
    for _ in range(reps):
        det.detectMarkers(img)
    return 1000 * (time.perf_counter() - t) / reps


def experiment_R1(photos=30):
    jobs = [(k, s, photos, wh) for k, s in LANDMARKS for wh in RES]
    jobs += [("cube", 0.10, photos, wh, BLUR_PX * wh[0] / 640) for wh in RES
             if wh[0] > 640]
    with N.pool() as p:
        res = p.map(S._fix_sweep, jobs, chunksize=1)
    s1 = {}
    path = os.path.join("results/scale/scale_results.json")
    if os.path.exists(path):
        s1 = json.load(open(path))["S1_fix_vs_distance_and_size"]["cube"]
    out, curves = {}, {}
    for job, m in zip(jobs, res):
        k, s, _, wh = job[:4]
        lens = len(job) > 4
        name = f"{k} {100 * s:g} cm | {res_label(wh)}" + (" | lens-limited" if lens else "")
        curves[name] = m
        seen = np.nonzero(m["detect"] >= 50)[0]
        d = {"kind": k, "size_m": s, "width_px": wh[0], "lens_limited": lens,
             "r_detect90_m": S.reach(m["detect"] >= 90),
             "r_fix10cm_m": S.reach((m["detect"] >= 50) &
                                    (np.nan_to_num(m["median"], nan=1e9) <= 10)),
             "r_min50_m": float(S.DISTS[seen[0]]) if len(seen) else np.inf,
             "median_err_cm_at": {f"{S.DISTS[i]:.2f} m": float(m["median"][i])
                                  for i in range(len(S.DISTS))}}
        if k == "cube" and s1:
            # Cube edge that reaches as far at 640 x 480 (log-log interpolation).
            edges = np.array([float(e) for e in s1])
            reach = np.array([s1[e]["r_fix10cm_m"] for e in s1])
            d["equivalent_cube_edge_at_640_cm"] = 100 * float(np.exp(np.interp(
                np.log(d["r_fix10cm_m"]), np.log(reach), np.log(edges),
                left=np.nan, right=np.nan)))
        out[name] = d
    out["detect_time_ms_per_frame"] = {res_label(wh): detect_time_ms(wh) for wh in RES}
    plot_R1(out, curves)
    return out


def plot_R1(out, curves):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = S.ramp(len(RES))
    for wh, col in zip(RES, cols):
        m = curves[f"cube 10 cm | {res_label(wh)}"]
        axes[0].plot(S.DISTS, m["detect"], color=col, lw=2, label=res_label(wh))
        axes[1].plot(S.DISTS, m["median"], color=col, lw=2, label=res_label(wh))
        lens = f"cube 10 cm | {res_label(wh)} | lens-limited"
        if lens in curves:
            axes[1].plot(S.DISTS, curves[lens]["median"], color=col, lw=1.2, ls="--")
    for ax in axes[:2]:
        S.plain_logx(ax)
        ax.set_xlabel("Distance to the cube (m)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("10 cm cube: photos with a fix (%)", loc="left", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[1].set_yscale("log")
    plain_log(axes[1])
    axes[1].axhline(10, color=MUTED, lw=1, ls=":")
    axes[1].set_title("Median position error (cm); dashed = lens-limited",
                      loc="left", fontsize=10)
    ax = axes[2]
    ws = np.array([wh[0] for wh in RES])
    for k, s, mk in (("cube", 0.10, "o"), ("wall", 0.15, "s")):
        for key, ls, lab in (("r_detect90_m", "-", "detected in 90 %"),
                             ("r_fix10cm_m", "--", "median fix within 10 cm")):
            ys = [out[f"{k} {100 * s:g} cm | {res_label(wh)}"][key] for wh in RES]
            ax.plot(ws, ys, color=C["single"] if k == "cube" else C["multi"],
                    lw=2, ls=ls, marker=mk, ms=5,
                    label=f"{'10 cm cube' if k == 'cube' else '15 cm wall tag'}, {lab}")
    lens = [out[f"cube 10 cm | {res_label(wh)} | lens-limited"]["r_fix10cm_m"]
            for wh in RES if wh[0] > 640]
    ax.plot(ws[ws > 640], lens, "o", color=C["single"], mfc="none", ms=8, mew=1.5,
            label="10 cm cube, 10 cm fixes, lens-limited")
    ref = out[f"cube 10 cm | {res_label(RES[1])}"]["r_detect90_m"]
    ax.plot(ws, ref * ws / 640, color=MUTED, lw=1, ls=":", label="proportional to width")
    S.plain_logx(ax)
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xlabel("Image width (px), 70 deg lens")
    ax.set_title("Reach (m)", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("R1. Fix quality vs camera resolution", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "resolution_fix.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# R2. Reaching spots with one 10 cm cube at each resolution
# ---------------------------------------------------------------------------
def experiment_R2(R1, seeds=4, n_goals=10):
    jobs = [("cube", W, 0.10, seed, n_goals, wh) for W in ROOMS_W for wh in RES
            for seed in range(seeds)]
    jobs.sort(key=lambda j: -j[1] * j[5][0] ** 2)       # slowest first
    with N.pool() as p:
        runs = p.map(S._nav_run, jobs, chunksize=1)
    out = {"tolerance_m": S.TOL, "seeds": seeds, "goals_per_run": n_goals,
           "conditions": {}}
    spots = []
    for W in ROOMS_W:
        for wh in RES:
            r_fix = R1[f"cube 10 cm | {res_label(wh)}"]["r_fix10cm_m"]
            g = [r for run in runs if run["W"] == W and tuple(run["camera"]) == wh
                 for r in run["goals"]]
            e = np.array([r["err_m"] for r in g])
            out["conditions"][f"room {W:g} m | {res_label(wh)}"] = {
                "room_w_m": W, "width_px": wh[0], "r_fix10cm_m": r_fix,
                "farthest_spot_m": max(r["landmark_dist_m"] for r in g),
                "err_median_cm": 100 * np.median(e),
                "err_p90_cm": 100 * np.percentile(e, 90),
                "within_10cm_pct": 100 * np.mean(e <= S.TOL),
                "fresh_fix_on_arrival_pct": 100 * np.mean([r["confirmed"] for r in g]),
                "steps_per_m_median": float(np.median(
                    [r["steps"] / r["dist_m"] for r in g if r["dist_m"] > 0.3]))}
            spots += [{"W": W, "width_px": wh[0], "r_fix": r_fix,
                       "landmark_dist_m": r["landmark_dist_m"], "err_m": r["err_m"],
                       "confirmed": r["confirmed"], "steps": r["steps"]} for r in g]
    out["spots"] = spots
    plot_R2(out, R1)
    return out


def plot_R2(out, R1):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                             gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
    ax = axes[0]
    grid = np.array([[out["conditions"][f"room {W:g} m | {res_label(wh)}"]["within_10cm_pct"]
                      for wh in RES] for W in ROOMS_W])
    im = ax.imshow(grid, origin="lower", cmap=S.SEQ, vmin=0, vmax=100, aspect="auto")
    for i in range(len(ROOMS_W)):
        for j in range(len(RES)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else INK)
    ax.set_xticks(range(len(RES)))
    ax.set_xticklabels([f"{wh[0]}" for wh in RES])
    ax.set_yticks(range(len(ROOMS_W)))
    ax.set_yticklabels([f"{W:g} x {S.ASPECT * W:g}" for W in ROOMS_W])
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Room (m)")
    ax.grid(False)
    ax.set_title("Spots reached within 10 cm (%), one 10 cm cube", loc="left",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.outline.set_visible(False)

    ax = axes[1]
    model = S.blind_model()
    ax.plot([0] + model["distance_m"], [100] + model["within_tol_pct"], color=MUTED,
            lw=2, ls="--", label="Model: blind walk from the edge of reach")
    for wh, col in zip(RES, S.ramp(len(RES))):
        g = [x for x in out["spots"] if x["width_px"] == wh[0]]
        xs = np.array([x["landmark_dist_m"] - x["r_fix"] for x in g])
        ok = np.array([x["err_m"] <= S.TOL for x in g])
        qs = np.quantile(xs, np.linspace(0, 1, 6))
        cx, cy = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (xs >= a) & (xs <= b)
            if m.sum() >= 8:
                cx.append(np.median(xs[m]))
                cy.append(100 * ok[m].mean())
        ax.plot(cx, cy, color=col, lw=2, marker="o", ms=4, label=res_label(wh))
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Spot's distance beyond the cube's 10 cm fix reach (m)")
    ax.set_title("Reached within 10 cm (%), per spot", loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[2]
    t = R1["detect_time_ms_per_frame"]
    ax.bar(range(len(RES)), [t[res_label(wh)] for wh in RES], 0.6,
           color=C["single"], edgecolor=SURF)
    for j, wh in enumerate(RES):
        v = t[res_label(wh)]
        ax.text(j, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(RES)))
    ax.set_xticklabels([f"{wh[0]}" for wh in RES])
    ax.set_xlabel("Image width (px)")
    ax.set_title("Tag detection time per frame (ms, this PC, 1 core)", loc="left",
                 fontsize=10)
    fig.suptitle("R2. Reaching spots with one 10 cm cube at different camera "
                 "resolutions (biases learned online)", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "resolution_nav.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/resolution")
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    ap.add_argument("--only", nargs="*", default=["R1", "R2"])
    a = ap.parse_args()
    OUT = a.out
    N.WORKERS = a.workers
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "resolution_results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    if "R1" in a.only:
        print("R1_fix_vs_resolution ...", flush=True)
        results["R1_fix_vs_resolution"] = experiment_R1()
        with open(path, "w") as f:
            json.dump(results, f, indent=1, default=float)
    if "R2" in a.only:
        print("R2_reaching_spots ...", flush=True)
        results["R2_reaching_spots"] = experiment_R2(results["R1_fix_vs_resolution"])
        with open(path, "w") as f:
            json.dump(results, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}/")
