"""
scale_experiments.py - how small can the landmark be for a given room?

  python3 -m experiments.scale_experiments [output_dir] [--workers N] [--only S1 S2]
  (from the repo root; default output: results/scale)

Two landmark kinds, sized freely:
  cube : one cube (edge e, tags 0.7 e) in the middle of the room
  wall : one tag (side s) in the middle of each wall, centre 12.5 cm + s/2
         above the floor (so a 15 cm tag sits at 20 cm, as in the arena)
Rooms are W x 0.75 W.  The map is exact here: a landmark placed 1 deg off
would add ~1.7 cm of error per metre from it whatever its size, which
would blur the size effect.

S1. Fix quality vs distance and size: the robot stands at distances
    0.3-16 m from one landmark, facing it, and takes photos.  Detection
    rate and fix error vs distance; does it all scale with distance / tag
    size?  Gives each size's window of good fixes: r_min, the closest
    distance where it is detected in >= 50 % of photos (big tags no longer
    fit in the frame close up), and r_fix, the farthest distance where it
    is detected in >= 50 % of photos with median error <= 10 cm.
S2. Reaching spots for every room size x landmark size (random drifting
    robots, biases learned online, Navigator that does not bother looking
    when no landmark is in range).  Per spot: true arrival error, its
    distance from the nearest landmark, whether the robot had a fresh fix
    on arrival.  Where does accurate navigation break down, and does the
    break follow distance / size, or distance beyond the fix reach?
"""
import sys, os, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from tagsim import *
from nav import *
import experiments.nav_experiments as N
from experiments.run_experiments import plain_log, INK, INK2, MUTED, GRID, AXIS, SURF, C

OUT = "results/scale"                 # set from the command line in __main__

ROOMS_W = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
ASPECT = 0.75
START = np.array([0.4, 0.4, 0.0])       # a corner, clear of any cube
SIZES = {"cube": (0.06, 0.10, 0.15, 0.25, 0.40),     # cube edge (m)
         "wall": (0.10, 0.15, 0.25, 0.40)}            # wall tag side (m)
KIND_LABEL = {"cube": "One cube in the middle", "wall": "One tag on each wall"}
SIZE_LABEL = {"cube": "cube edge", "wall": "tag side"}
CUBE_TAG_FRAC = 0.7
DISTS = np.geomspace(0.3, 16.0, 18)
TOL = 0.10                            # "reached the spot" tolerance (m)
# Ordinal blue ramp (palette sequential steps 250-700), small -> large size.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def ramp(n):
    """n distinct steps of the ordinal ramp, light to dark."""
    return [RAMP[i] for i in np.linspace(0, len(RAMP) - 1, n).round().astype(int)]


def plain_logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))


def tag_side(kind, size):
    return CUBE_TAG_FRAC * size if kind == "cube" else size


def wall_tag_height(side):
    return 0.125 + side / 2


def scale_scene(kind, room, size, rng):
    """Scene dict as nav_experiments.make_scene, for any room and size."""
    W, D = room
    cam = Camera()
    K_est = perturb_intrinsics(cam.K, rng, 0.01, 3.0)
    cube = None
    if kind == "cube":
        cube = Cube(make_T(rot_z(np.radians(20)), [W / 2, D / 2, size / 2]),
                    size, CUBE_TAG_FRAC * size)
        tags = cube.tags()
        lms = [Landmark(np.array([W / 2, D / 2]), None,
                        tag_side(kind, size) * cam.f / N.DETECT_PX)]
    else:
        h = wall_tag_height(size)
        mids = [((W / 2, 0), (0, 1)), ((W, D / 2), (-1, 0)),
                ((W / 2, D), (0, -1)), ((0, D / 2), (1, 0))]
        tags = [Tag(i, np.array([x, y, h]), np.array([nx, ny, 0.0]), size)
                for i, ((x, y), (nx, ny)) in enumerate(mids)]
        lms = [Landmark(t.center[:2], t.normal[:2], size * cam.f / N.DETECT_PX)
               for t in tags]
    return {"tags": tags, "map_tags": tags, "cam": cam, "K_est": K_est,
            "W": W, "D": D, "landmarks": lms, "cube": cube}


def landmark_dist(kind, sc, p):
    """Distance from p to the nearest landmark that can be seen from p."""
    return min(np.linalg.norm(l.xy - p) for l in sc["landmarks"]
               if l.in_view_cone(p, 89.0))


# ---------------------------------------------------------------------------
# S1. Fix quality vs distance and size
# ---------------------------------------------------------------------------
def _fix_sweep(args):
    kind, size, photos = args
    rng = np.random.default_rng([11, int(1000 * size), kind == "cube"])
    cam = Camera()
    K_est = perturb_intrinsics(cam.K, rng, 0.01, 3.0)
    if kind == "cube":
        tags = Cube(make_T(rot_z(np.radians(20)), [0, 0, size / 2]), size,
                    CUBE_TAG_FRAC * size).tags()
    else:
        tags = [Tag(0, np.array([0, 0, wall_tag_height(size)]),
                    np.array([0, 1.0, 0]), size)]
    det = make_detector()
    out = {k: [] for k in ("detect", "median", "p90", "flip")}
    for d in DISTS:
        hit, pos, yaw = 0, [], []
        for _ in range(photos):
            a = rng.uniform(-np.pi, np.pi) if kind == "cube" else \
                np.radians(rng.uniform(-60, 60))
            p = (-d * np.cos(a), -d * np.sin(a)) if kind == "cube" else \
                (d * np.sin(a), d * np.cos(a))
            face = np.arctan2(-p[1], -p[0]) + np.radians(rng.uniform(-10, 10))
            r = measure((p[0], p[1], face), tags, cam, N.STAND, rng, detector=det,
                        K_est=K_est)
            if r.n_tags:
                hit += 1
                pos.append(r.err["floor"][0])
                yaw.append(r.err["floor"][1])
        out["detect"].append(100 * hit / photos)
        out["median"].append(100 * np.median(pos) if pos else np.nan)
        out["p90"].append(100 * np.percentile(pos, 90) if pos else np.nan)
        out["flip"].append(100 * np.mean(np.array(yaw) > 10) if yaw else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def reach(curve_ok):
    """Distance up to which curve_ok holds, starting from the first
    distance where it holds (big tags are cut off when very close).
    Geometric midpoint between the last good and first bad distance."""
    idx = np.nonzero(curve_ok)[0]
    if not len(idx):
        return 0.0
    i = idx[0]
    while i + 1 < len(DISTS) and curve_ok[i + 1]:
        i += 1
    return float(DISTS[i] if i + 1 == len(DISTS) else np.sqrt(DISTS[i] * DISTS[i + 1]))


def experiment_S1(photos=40):
    jobs = [(k, s, photos) for k in SIZES for s in SIZES[k]]
    with N.pool() as p:
        res = dict(zip([(k, s) for k, s, _ in jobs], p.map(_fix_sweep, jobs)))
    out = {}
    for (k, s), m in res.items():
        side = tag_side(k, s)
        r_det = reach(m["detect"] >= 90)
        seen = np.nonzero(m["detect"] >= 50)[0]
        r_min = float(DISTS[seen[0]] if seen[0] == 0 else
                      np.sqrt(DISTS[seen[0] - 1] * DISTS[seen[0]])) if len(seen) else np.inf
        r_fix = reach((m["detect"] >= 50) & (np.nan_to_num(m["median"], nan=1e9) <= 10))
        out.setdefault(k, {})[f"{s:g}"] = {
            "tag_side_m": side, "r_min50_m": r_min, "r_detect90_m": r_det,
            "r_fix10cm_m": r_fix,
            "r_detect90_over_side": r_det / side, "r_fix10cm_over_side": r_fix / side,
            "by_distance": {f"{d:.2f}": {"detect_pct": m["detect"][i],
                                         "median_cm": m["median"][i],
                                         "p90_cm": m["p90"][i],
                                         "flip_pct": m["flip"][i]}
                            for i, d in enumerate(DISTS)}}
    plot_S1(res)
    return out


def plot_S1(res):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for row, kind in enumerate(SIZES):
        for s, col in zip(SIZES[kind], ramp(len(SIZES[kind]))):
            m, side = res[(kind, s)], tag_side(kind, s)
            lab = f"{100 * s:g} cm {SIZE_LABEL[kind]}"
            axes[row, 0].plot(DISTS, m["detect"], color=col, lw=2, label=lab)
            axes[row, 1].plot(DISTS, m["median"], color=col, lw=2, label=lab)
            axes[row, 2].plot(DISTS / side, m["median"] / (100 * side), color=col,
                              lw=2, label=lab)
        for ax in axes[row]:
            plain_logx(ax)
        axes[row, 0].set_ylim(0, 105)
        axes[row, 0].set_xlabel("Distance to landmark (m)")
        axes[row, 0].set_title(f"{KIND_LABEL[kind]}: photos with a fix (%)",
                               loc="left", fontsize=10)
        axes[row, 1].set_yscale("log")
        plain_log(axes[row, 1])
        axes[row, 1].axhline(10, color=MUTED, lw=1, ls=":")
        axes[row, 1].set_xlabel("Distance to landmark (m)")
        axes[row, 1].set_title("Median position error (cm)", loc="left", fontsize=10)
        axes[row, 2].set_yscale("log")
        plain_log(axes[row, 2])
        axes[row, 2].set_xlabel("Distance / tag side")
        axes[row, 2].set_title("Same, in units of tag side (collapses if it "
                               "only scales)", loc="left", fontsize=10)
        axes[row, 0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("S1. Fix quality vs distance for different landmark sizes "
                 "(640 x 480 camera, 70 deg field of view)", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scale_fix_vs_distance.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# S2. Reaching spots across room size x landmark size
# ---------------------------------------------------------------------------
def _nav_run(args):
    kind, W, size, seed, n_goals = args
    D = ASPECT * W
    rng = np.random.default_rng([seed, int(10 * W), int(1000 * size)])
    sc = scale_scene(kind, (W, D), size, np.random.default_rng(500 + seed))
    motors = random_motors(np.random.default_rng(600 + seed))
    drift_rng = np.random.default_rng(700 + seed)
    centre = np.array([W / 2, D / 2])
    clear = 0.25 + size / 2 if kind == "cube" else 0.0
    goals = N.goal_sequence(np.random.default_rng(800 + seed), n_goals, W, D,
                            clear=clear, centre=centre, start=START)
    records, path, ekf = N.run_goals(
        sc, goals, motors, True, rng, drift_rng, lambda d: int(200 + 60 * d),
        nav_kw={"look_only_in_range": True}, start=START)
    for g, r in zip(goals, records):
        r["landmark_dist_m"] = landmark_dist(kind, sc, g)
    return {"kind": kind, "W": W, "size": size, "seed": seed, "goals": records}


def experiment_S2(S1, seeds=5, n_goals=10):
    jobs = [(k, W, s, seed, n_goals) for k in SIZES for W in ROOMS_W
            for s in SIZES[k] for seed in range(seeds)]
    # Longest runs first so the pool finishes evenly.
    jobs.sort(key=lambda j: -j[1])
    with N.pool() as p:
        runs = p.map(_nav_run, jobs, chunksize=1)
    out = {"tolerance_m": TOL, "seeds": seeds, "goals_per_run": n_goals,
           "conditions": {}}
    for k in SIZES:
        for W in ROOMS_W:
            for s in SIZES[k]:
                g = [r for run in runs if (run["kind"], run["W"], run["size"]) ==
                     (k, W, s) for r in run["goals"]]
                e = np.array([r["err_m"] for r in g])
                r_fix = S1[k][f"{s:g}"]["r_fix10cm_m"]
                r_min = S1[k][f"{s:g}"]["r_min50_m"]
                far = max(r["landmark_dist_m"] for r in g)
                out["conditions"][f"{k} | room {W:g} m | {SIZE_LABEL[k]} {100 * s:g} cm"] = {
                    "kind": k, "room_w_m": W, "size_m": s, "r_min50_m": r_min,
                    "r_fix10cm_m": r_fix,
                    "room_over_tag_side": W / tag_side(k, s),
                    "farthest_spot_from_landmark_m": far,
                    "err_median_cm": 100 * np.median(e),
                    "err_p90_cm": 100 * np.percentile(e, 90),
                    "within_10cm_pct": 100 * np.mean(e <= TOL),
                    "within_20cm_pct": 100 * np.mean(e <= 2 * TOL),
                    "fresh_fix_on_arrival_pct": 100 * np.mean([r["confirmed"] for r in g]),
                    "timeouts_pct": 100 * np.mean([not r["arrived"] for r in g]),
                    "steps_per_m_median": float(np.median(
                        [r["steps"] / r["dist_m"] for r in g if r["dist_m"] > 0.3]))}
    per_goal = [{"kind": run["kind"], "W": run["W"], "size": run["size"],
                 "r_fix": S1[run["kind"]][f"{run['size']:g}"]["r_fix10cm_m"], **r}
                for run in runs for r in run["goals"]]
    out["by_distance_beyond_reach"] = collapse_table(per_goal)
    out["blind_walk_model"] = blind_model()
    out["spots"] = [{k: g[k] for k in ("kind", "W", "size", "r_fix",
                                       "landmark_dist_m", "err_m", "confirmed",
                                       "arrived", "steps")} for g in per_goal]
    plot_S2_grid(out)
    plot_S2_collapse(out["spots"], out["blind_walk_model"])
    return out


def blind_model(trials=400):
    """Experiment 2's motion model (biases learned, turn then walk blind):
    share of walks still within TOL after each distance."""
    pos = np.array([N.blind_trial(seed, 100)[0]["learned"] for seed in range(trials)])
    return {"distance_m": list(0.1 * np.arange(1, pos.shape[1] + 1)),
            "within_tol_pct": list(100 * np.mean(pos <= TOL, axis=0))}


def collapse_table(per_goal):
    """Success rate by how far beyond the landmark's fix reach a spot is."""
    bins = [(-99, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2),
            (2, 3), (3, 99)]
    out = {}
    for k in SIZES:
        for a, b in bins:
            sel = [g for g in per_goal if g["kind"] == k and
                   a <= g["landmark_dist_m"] - g["r_fix"] < b]
            if len(sel) >= 10:
                e = np.array([g["err_m"] for g in sel])
                out.setdefault(k, {})[f"{a:g} to {b:g} m"] = {
                    "spots": len(sel), "within_10cm_pct": 100 * np.mean(e <= TOL),
                    "err_median_cm": 100 * np.median(e),
                    "fresh_fix_on_arrival_pct": 100 * np.mean([g["confirmed"] for g in sel])}
    return out


def plot_S2_grid(out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={"width_ratios": [5, 4]})
    for ax, k in zip(axes, SIZES):
        sizes = SIZES[k]
        grid = np.array([[out["conditions"][f"{k} | room {W:g} m | {SIZE_LABEL[k]} "
                                            f"{100 * s:g} cm"]["within_10cm_pct"]
                          for s in sizes] for W in ROOMS_W])
        im = ax.imshow(grid, origin="lower", cmap=SEQ, vmin=0, vmax=100,
                       aspect="auto")
        for i, W in enumerate(ROOMS_W):
            for j, s in enumerate(sizes):
                c = out["conditions"][f"{k} | room {W:g} m | {SIZE_LABEL[k]} {100 * s:g} cm"]
                v = c["within_10cm_pct"]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 55 else INK)
                # Predicted break: farthest spot beyond fix reach + ~1 m blind,
                # or (single landmark) nowhere far enough away to see it whole.
                far = c["farthest_spot_from_landmark_m"]
                if far > c["r_fix10cm_m"] + 1.1 or (k == "cube" and far < c["r_min50_m"]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               ec=C["multi"], lw=1.5, ls="--"))
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"{100 * s:g}" for s in sizes])
        ax.set_yticks(range(len(ROOMS_W)))
        ax.set_yticklabels([f"{W:g} x {ASPECT * W:g}" for W in ROOMS_W])
        ax.set_xlabel(f"{SIZE_LABEL[k].capitalize()} (cm)")
        ax.set_ylabel("Room (m)")
        ax.grid(False)
        ax.set_title(f"{KIND_LABEL[k]}", loc="left")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cb.set_label(f"Spots reached within {100 * TOL:.0f} cm (%)", color=INK2)
    cb.outline.set_visible(False)
    fig.text(0.01, -0.06, "Dashed orange outline: predicted to fail - the farthest "
             "spot is more than ~1.1 m (blind-walking reach) beyond where the "
             "landmark still gives 10 cm fixes,\nor (single cube) no spot is far "
             "enough from it for it to fit in the camera's view.", color=INK2,
             fontsize=9)
    fig.suptitle("S2. Reaching spots: room size vs landmark size (biases learned "
                 "online)", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.savefig(os.path.join(OUT, "scale_room_vs_landmark.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_S2_collapse(per_goal, model):
    """Per spot: is the right variable distance / size, or distance beyond
    the landmark's fix reach?"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    for row, k in enumerate(SIZES):
        axes[row, 1].plot([0] + model["distance_m"], [100] + model["within_tol_pct"],
                          color=MUTED, lw=2, ls="--",
                          label="Model: blind walk from the edge of reach")
        for s, col in zip(SIZES[k], ramp(len(SIZES[k]))):
            g = [x for x in per_goal if x["kind"] == k and x["size"] == s]
            for ax, xs in ((axes[row, 0], [x["landmark_dist_m"] / tag_side(k, s) for x in g]),
                           (axes[row, 1], [x["landmark_dist_m"] - x["r_fix"] for x in g])):
                xs = np.array(xs)
                ok = np.array([x["err_m"] <= TOL for x in g])
                qs = np.quantile(xs, np.linspace(0, 1, 7))
                cx, cy = [], []
                for a, b in zip(qs[:-1], qs[1:]):
                    m = (xs >= a) & (xs <= b)
                    if m.sum() >= 8:
                        cx.append(np.median(xs[m]))
                        cy.append(100 * ok[m].mean())
                ax.plot(cx, cy, color=col, lw=2, marker="o", ms=4,
                        label=f"{100 * s:g} cm {SIZE_LABEL[k]}")
        plain_logx(axes[row, 0])
        axes[row, 0].set_xlabel("Spot's distance from landmark / tag side")
        axes[row, 1].set_xlabel("Spot's distance beyond the landmark's 10 cm fix reach (m)")
        axes[row, 1].axvline(0, color=MUTED, lw=1, ls=":")
        for ax in axes[row]:
            ax.set_ylim(-3, 103)
            ax.set_ylabel(f"Reached within {100 * TOL:.0f} cm (%)")
        axes[row, 0].set_title(f"{KIND_LABEL[k]}: vs distance / size", loc="left",
                               fontsize=10)
        axes[row, 1].set_title(f"{KIND_LABEL[k]}: vs distance beyond fix reach",
                               loc="left", fontsize=10)
        axes[row, 0].legend(loc="lower left", fontsize=8.5)
        axes[row, 1].legend(handles=axes[row, 1].get_lines()[:1], loc="lower left",
                            fontsize=8.5)
    fig.suptitle("S2. Which distance explains the breakdown? (each point: one "
                 "sixth of a size's spots)", x=0.01, ha="left", color=INK,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scale_breakdown.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/scale")
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    ap.add_argument("--only", nargs="*", default=["S1", "S2"])
    a = ap.parse_args()
    OUT = a.out
    N.WORKERS = a.workers
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "scale_results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    if "S1" in a.only:
        print("S1_fix_vs_distance_and_size ...", flush=True)
        results["S1_fix_vs_distance_and_size"] = experiment_S1()
        with open(path, "w") as f:
            json.dump(results, f, indent=1, default=float)
    if "S2" in a.only:
        print("S2_reaching_spots ...", flush=True)
        results["S2_reaching_spots"] = experiment_S2(results["S1_fix_vs_distance_and_size"])
        with open(path, "w") as f:
            json.dump(results, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}/")
