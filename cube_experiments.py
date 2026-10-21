"""
cube_experiments.py - can the robot find a tagged cube?

  python3 cube_experiments.py [output_dir]

A 10 cm cube with a 7 cm AprilTag centred on each face (ids 100-105) is
lying on a random face, at a random spot and yaw, inside the arena of
run_experiments.py.  The robot knows the cube's geometry but not its pose.

D. Single photos: the robot stands at a random distance from the cube,
   roughly facing it.  The cube's pose relative to the robot is estimated
   with plain 6-DOF PnP and with the floor constraint (cube lying flat),
   then placed on the map using the robot's own tag fix from the same
   photo.  Error vs distance, and how often the orientation comes out
   wrong.  Plus a few variations (cube size, walking blur, a cube that is
   not lying on the floor).
E. Consistency while moving: the robot walks the experiment-C loop with
   its bad motors, turns to look at the cube at every waypoint once it has
   seen it, and fuses all placements.  Do the per-photo placements agree
   from different viewpoints, and does the fused estimate settle?  One
   run in detail, then several random cube placements.
Also saves an annotated example camera frame.

Detection rates are "detected_pct_of_in_frame": photos in which at least
one cube tag was decoded, out of photos in which the cube was at least
partly inside the image (checked geometrically with the true camera pose).
Photos with the cube out of frame are counted separately, never as misses.
"""
import sys, os, json, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation
from tagsim import *
from run_experiments import (realistic_setup, simulate_run, plain_log, INK,
                             INK2, MUTED, GRID, AXIS, SURF, C)

OUT = "results"          # set from the command line in __main__

CUBE_C = {"pnp": C["multi"], "floor": C["single"], "world": C["floor"]}
FUSED = C["multi"]                     # fused cube estimate in experiment E
CUBE_LABEL = {"pnp": "6-DOF PnP, relative to robot",
              "floor": "Floor constraint, relative to robot",
              "world": "Floor constraint, on the map (robot's own tag fix)"}
CUBE_XLIM, CUBE_YLIM = (0.6, 3.4), (0.6, 2.4)     # where cubes are placed
WAYPOINTS = [(2.0, 0.7), (3.3, 0.7), (3.3, 1.5), (3.3, 2.3),
             (2.0, 2.3), (0.7, 2.3), (0.7, 1.5), (0.7, 0.7)]
POS_FAIL_M, ROT_FAIL_DEG = 0.10, 15.0


def random_cube(rng, edge=0.10, tag_size=0.07, xlim=CUBE_XLIM, ylim=CUBE_YLIM):
    return Cube(random_cube_pose(rng, xlim, ylim, edge), edge, tag_size)


def facing_pose(rng, cube, W, D, d_range=(0.3, 2.8), off_deg=25, margin=0.25):
    """Robot pose at a random distance from the cube, roughly facing it."""
    c = cube.T[:2, 3]
    while True:
        d, a = rng.uniform(*d_range), rng.uniform(-np.pi, np.pi)
        p = c - d * np.array([np.cos(a), np.sin(a)])
        if margin < p[0] < W - margin and margin < p[1] < D - margin:
            return (p[0], p[1], wrap(a + np.radians(rng.uniform(-off_deg, off_deg))))


def cube_summary(results, methods=("pnp", "floor")):
    n = len(results)
    hit = [r for r in results if r.cube_faces > 0]
    inf = [r for r in results if r.cube_in_frame in ("full", "partial")]
    full = [r for r in results if r.cube_in_frame == "full"]
    s = {"frames": n,
         "cube_in_frame_pct": 100 * len(inf) / n,
         "cube_fully_in_frame_pct": 100 * len(full) / n,
         "detected_pct_of_in_frame": 100 * np.mean([r.cube_faces > 0 for r in inf]),
         "detected_pct_of_fully_in_frame":
             100 * np.mean([r.cube_faces > 0 for r in full]),
         "two_faces_pct_of_detected":
             100 * np.mean([r.cube_faces >= 2 for r in hit])}
    rows = [(m, "rel", m) for m in methods] + [("floor", "world", "world")]
    for m, kind, key in rows:
        e = np.array([(r.cube_err_rel if kind == "rel" else r.cube_err_world)[m]
                      for r in hit if m in (r.cube_err_rel if kind == "rel"
                                            else r.cube_err_world)])
        if len(e) == 0:
            continue
        s[key] = {"pos_median_cm": 100 * np.median(e[:, 0]),
                  "pos_p90_cm": 100 * np.percentile(e[:, 0], 90),
                  "rot_median_deg": np.median(e[:, 1]),
                  "rot_p90_deg": np.percentile(e[:, 1], 90),
                  f"pos_fail_pct (>{100 * POS_FAIL_M:.0f} cm)":
                      100 * np.mean(e[:, 0] > POS_FAIL_M),
                  f"rot_fail_pct (>{ROT_FAIL_DEG:.0f} deg)":
                      100 * np.mean(e[:, 1] > ROT_FAIL_DEG)}
    return s


# ---------------------------------------------------------------------------
# Example frame
# ---------------------------------------------------------------------------
CUBE_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def example_frame():
    rng = np.random.default_rng(11)
    tags, map_tags, cam, K_est, (W, D) = realistic_setup(rng)
    cube = Cube(make_T(rot_z(np.radians(70)) @ rest_rotation(0, 1),
                       [2.3, 1.4, 0.05]))
    pose = (1.55, 0.95, np.radians(28))
    r, img, det = measure(pose, tags, cam, Degradation(), rng, K_est=K_est,
                          map_tags=map_tags, cube=cube, return_image=True)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blue, orange = (214, 120, 42), (52, 104, 235)
    for tag, uv in det:
        col = orange if tag.id in cube.ids else blue
        cv2.polylines(vis, [uv.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        cv2.putText(vis, f"id {tag.id}", tuple(uv[0].astype(int) + [0, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    # Estimated cube (floor method) drawn back into the image, through the
    # camera wobble estimated along with it.
    cdet = [d for d in det if d[0].id in cube.ids]
    _, wob = estimate_object_floor(cdet, K_est, None, cam,
                                   estimate_object_pnp(cdet, K_est, None),
                                   cube.edge / 2, return_wobble=True)
    T_cam_cube = inv_T(cam.T_robot_cam(*wob)) @ r.cube_rel["floor"]
    pts = cube.corners(T_cam_cube)
    uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), cam.K, None)
    uv = uv.reshape(-1, 2)
    for a, b in CUBE_EDGES:
        cv2.line(vis, tuple(np.round(uv[a]).astype(int)),
                 tuple(np.round(uv[b]).astype(int)), (60, 200, 60), 1, cv2.LINE_AA)
    T_est = robot_T(*r.est["floor"]) @ r.cube_rel["floor"]
    e_pos, e_rot = r.cube_err_world["floor"]
    lines = [f"cube true  x={cube.T[0, 3]:.3f} y={cube.T[1, 3]:.3f}",
             f"cube est   x={T_est[0, 3]:.3f} y={T_est[1, 3]:.3f}   "
             f"err {100 * e_pos:.1f} cm, {e_rot:.1f} deg",
             f"{r.cube_faces} cube faces, {r.n_tags} wall tags; green = estimated cube"]
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (10, 425 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, "cube_frame.png"), vis)


# ---------------------------------------------------------------------------
# D. Single photos
# ---------------------------------------------------------------------------
def run_D(N, rng, cube_kw=None, deg=None, floating=False):
    tags, map_tags, cam, K_est, (W, D) = realistic_setup(rng)
    det = make_detector()
    out = []
    for _ in range(N):
        cube = random_cube(rng, **(cube_kw or {}))
        if floating:          # any orientation, centre 5-30 cm above the floor
            R = Rotation.random(random_state=rng).as_matrix()
            cube.T = make_T(R, [cube.T[0, 3], cube.T[1, 3], rng.uniform(0.05, 0.30)])
        out.append(measure(facing_pose(rng, cube, W, D), tags, cam,
                           deg or Degradation(), rng, detector=det, K_est=K_est,
                           map_tags=map_tags, cube=cube, cube_floor=not floating))
    return out


def experiment_D(N=1500):
    res = run_D(N, np.random.default_rng(21))
    hit = [r for r in res if r.cube_faces > 0]
    d = np.array([r.cube_dist for r in hit])
    bins = np.arange(0.3, 2.81, 0.25)
    centers = (bins[:-1] + bins[1:]) / 2
    series = [("pnp", lambda r: r.cube_err_rel.get("pnp")),
              ("floor", lambda r: r.cube_err_rel.get("floor")),
              ("world", lambda r: r.cube_err_world.get("floor"))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, k, unit, title in ((axes[0], 0, 100, "Median position error (cm)"),
                               (axes[1], 1, 1, "Median orientation error (deg)")):
        for key, get in series:
            ok = np.array([get(r) is not None for r in hit])
            e = np.array([get(r)[k] if get(r) is not None else np.nan
                          for r in hit]) * unit
            med = []
            for a, b in zip(bins[:-1], bins[1:]):
                m = ok & (d >= a) & (d < b)
                med.append(np.median(e[m]) if m.sum() >= 15 else np.nan)
            ax.plot(centers, med, color=CUBE_C[key], lw=2, marker="o", ms=4,
                    label=CUBE_LABEL[key])
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlabel("Distance from camera to cube (m)")
        ax.set_title(title, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left",
               bbox_to_anchor=(0.01, 0.0), ncol=3, fontsize=9)

    # Orientation failures: with one face in view the tag's mirror ambiguity
    # flips the cube's yaw; with two faces it never happened.
    ax = axes[2]
    one = [r for r in hit if r.cube_faces == 1]
    d1 = np.array([r.cube_dist for r in one])
    for key in ("pnp", "floor"):
        bad = np.array([r.cube_err_rel[key][1] > ROT_FAIL_DEG for r in one])
        frac = [100 * bad[(d1 >= a) & (d1 < b)].mean()
                if np.sum((d1 >= a) & (d1 < b)) >= 15 else np.nan
                for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(centers, frac, color=CUBE_C[key], lw=2, marker="o", ms=4)
    two = [r for r in hit if r.cube_faces >= 2]
    n_bad = sum(r.cube_err_rel[k][1] > ROT_FAIL_DEG for r in two
                for k in ("pnp", "floor"))
    ax.text(0.03, 0.95, f"Two faces in view: {n_bad} failures\n"
            f"in {len(two)} photos (either method)", transform=ax.transAxes,
            va="top", color=INK2, fontsize=9)
    ax.set_ylim(0, None)
    ax.set_xlabel("Distance from camera to cube (m)")
    ax.set_title(f"One face in view: orientation off by > {ROT_FAIL_DEG:.0f} deg (%)",
                 loc="left")
    fig.suptitle(f"Locating a 10 cm tagged cube from single photos "
                 f"({len(hit)} photos with the cube detected)",
                 x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "cube_error_vs_distance.png"), dpi=150)
    plt.close(fig)

    s = cube_summary(res)
    s["by_faces_seen"] = {f"{nf}": cube_summary([r for r in res if r.cube_faces == nf])
                          for nf in (1, 2)}
    s["by_distance"] = {}
    for a, b in ((0.3, 1.0), (1.0, 1.8), (1.8, 2.8)):
        s["by_distance"][f"{a}-{b} m"] = cube_summary(
            [r for r in hit if a <= r.cube_dist < b])
    return s


def experiment_D2(N=500):
    scenarios = {
        "baseline (10 cm cube, 7 cm tags)": {},
        "small cube (6 cm, 4 cm tags)": {"cube_kw": {"edge": 0.06, "tag_size": 0.04}},
        "large cube (15 cm, 11 cm tags)": {"cube_kw": {"edge": 0.15, "tag_size": 0.11}},
        "photo while walking (5 px motion blur)": {"deg": Degradation(motion_px=5)},
        "cube not on the floor (any orientation, PnP only)": {"floating": True},
    }
    out = {}
    for name, kw in scenarios.items():
        res = run_D(N, np.random.default_rng(321), **kw)
        out[name] = cube_summary(res, ("pnp",) if kw.get("floating") else
                                 ("pnp", "floor"))
        m = out[name].get("world") or out[name].get("pnp", {})
        print(f"  {name:52s} detected "
              f"{out[name]['detected_pct_of_in_frame']:5.1f}% of in-frame  "
              f"{'map' if 'world' in out[name] else 'rel'} median "
              f"{m.get('pos_median_cm', np.nan):5.1f} cm  "
              f"{m.get('rot_median_deg', np.nan):5.1f} deg  rot fail "
              f"{m.get(f'rot_fail_pct (>{ROT_FAIL_DEG:.0f} deg)', np.nan):4.1f}%",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# E. Consistency while moving
# ---------------------------------------------------------------------------
def cube_run(seed, steps):
    rng = np.random.default_rng(1000 + seed)
    # Inside the loop, at least ~0.4 m from the waypoint path.
    cube = random_cube(rng, xlim=(1.3, 2.7), ylim=(1.1, 1.9))
    log, ctx = simulate_run(steps, seed=5 + seed, cube=cube, waypoints=WAYPOINTS,
                            look_at_cube=True)
    return log, ctx


def run_stats(log, ctx):
    seen = log["cube_faces"] > 0
    full = log["cube_in_frame"] == "full"
    inf = full | (log["cube_in_frame"] == "partial")
    acc = log["cube_accepted"]
    ew, er, ef = log["cube_err_world"], log["cube_err_rel"], log["cube_err_fused"]
    P = np.array([T[:2, 3] for T, a in zip(log["cube_T"], acc) if a])
    first = int(np.argmax(seen)) if seen.any() else None
    true_xy = ctx["cube"].T[:2, 3]
    dmin = np.min(np.hypot(*(log["true"][:, :2] - true_xy).T))
    return {
        "photos": len(seen),
        "photos_cube_in_frame": int(inf.sum()),
        "photos_cube_fully_in_frame": int(full.sum()),
        "photos_cube_detected": int(seen.sum()),
        "detected_pct_of_in_frame": 100 * seen[inf].mean(),
        "detected_pct_of_fully_in_frame": 100 * seen[full].mean(),
        "first_seen_step": first,
        "rejected_by_gate": int((seen & ~acc).sum()),
        "per_photo_map_err_median_cm": 100 * np.nanmedian(ew[seen, 0]),
        "per_photo_map_err_p90_cm": 100 * np.nanpercentile(ew[seen, 0], 90),
        "per_photo_map_err_max_cm": 100 * np.nanmax(ew[seen, 0]),
        "per_photo_rel_err_median_cm": 100 * np.nanmedian(er[seen, 0]),
        "per_photo_rot_err_median_deg": np.nanmedian(ew[seen, 1]),
        "per_photo_rot_fail_pct": 100 * np.mean(ew[seen, 1] > ROT_FAIL_DEG),
        "placement_spread_rms_cm": 100 * np.sqrt(np.mean(np.sum(
            (P - P.mean(0)) ** 2, 1))) if len(P) else np.nan,
        "fused_final_err_cm": 100 * ef[-1, 0], "fused_final_rot_deg": ef[-1, 1],
        "fused_max_err_after_10_photos_cm": 100 * np.nanmax(
            ef[np.cumsum(seen) >= 10, 0]) if seen.sum() >= 10 else np.nan,
        "closest_approach_m": dmin,
    }


def experiment_E(steps=320, n_runs=8):
    log, ctx = cube_run(0, steps)
    detail = run_stats(log, ctx)
    plot_E(log, ctx)
    runs = []
    for seed in range(1, n_runs):
        lg, cx = cube_run(seed, steps)
        runs.append(run_stats(lg, cx))
        st = runs[-1]
        print(f"  run {seed}: cube in frame in {st['photos_cube_in_frame']} of "
              f"{st['photos']} photos, detected in {st['photos_cube_detected']} "
              f"({st['detected_pct_of_in_frame']:.0f}%), per-photo median "
              f"{st['per_photo_map_err_median_cm']:.1f} cm, spread "
              f"{st['placement_spread_rms_cm']:.1f} cm, fused final "
              f"{st['fused_final_err_cm']:.1f} cm / {st['fused_final_rot_deg']:.1f} deg",
              flush=True)
    return {"detail_run": detail, "other_runs": runs}


def plot_E(log, ctx):
    cube, tags, W, D = ctx["cube"], ctx["tags"], ctx["W"], ctx["D"]
    T, KF = log["true"], log["kf"]
    seen = log["cube_faces"] > 0
    acc = log["cube_accepted"]
    ew, ef = log["cube_err_world"], log["cube_err_fused"]
    steps = np.arange(1, len(T) + 1)
    foot = lambda Tc: (Tc[:3, :3] @ (cube.edge / 2 * np.array(
        [[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])).T).T[:, :2] + Tc[:2, 3]
    # Front-face direction marker so a yaw error is visible.
    front = lambda Tc: np.array([Tc[:2, 3], Tc[:2, 3] + 0.08 * Tc[:2, 0]
                                 / max(np.linalg.norm(Tc[:2, 0]), 1e-9)])

    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.42, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0, 0), W, D, fill=False, ec=AXIS, lw=1.5))
    for t in tags:
        ax.plot(*t.center[:2], "s", color=INK2, ms=4)
    ax.plot(T[:, 0], T[:, 1], color=INK, lw=1.2, alpha=0.6)
    P = T[seen]
    ax.plot(P[:, 0], P[:, 1], "o", color=CUBE_C["floor"], ms=4, zorder=4)
    for p in P[::3]:
        ax.plot([p[0], cube.T[0, 3]], [p[1], cube.T[1, 3]], color=CUBE_C["floor"],
                lw=0.5, alpha=0.35, zorder=3)
    ax.add_patch(Polygon(foot(cube.T), fc=INK, ec=INK, zorder=5))
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, W + 0.3)
    ax.set_ylim(-0.3, D + 0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Where the cube was photographed from", loc="left")
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.2, alpha=0.6, label="True path"),
                       Line2D([], [], marker="o", ls="", color=CUBE_C["floor"],
                              label="Photo showing the cube"),
                       Line2D([], [], marker="s", ls="", color=INK, label="Cube")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)

    # Zoom on the cube: every placement plus the fused one.
    ax = fig.add_subplot(gs[0, 1])
    c = cube.T[:2, 3]
    r = 0.14
    ax.add_patch(Polygon(foot(cube.T), fill=False, ec=INK, lw=2, zorder=3))
    ax.plot(*front(cube.T).T, color=INK, lw=2, zorder=3)
    Ts = [Tc for Tc in log["cube_T"] if Tc is not None]
    a = acc[seen]
    pts = np.array([Tc[:2, 3] for Tc in Ts])
    ax.plot(pts[a, 0], pts[a, 1], "o", color=CUBE_C["floor"], ms=4, alpha=0.6,
            mec="none", zorder=4)
    if (~a).any():
        ax.plot(pts[~a, 0], pts[~a, 1], "x", color=INK2, ms=7, mew=2,
                zorder=4)
    Tf = ctx["cube_tracker"].T
    ax.add_patch(Polygon(foot(Tf), fill=False, ec=FUSED, lw=2, ls="--",
                         zorder=5))
    ax.plot(*front(Tf).T, color=FUSED, lw=2, ls="--", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_xlabel("x (m)")
    ax.set_title("Cube placements on the map (zoomed)", loc="left")
    handles = [Line2D([], [], color=INK, lw=2, label="True cube (line = face 0)"),
               Line2D([], [], marker="o", ls="", color=CUBE_C["floor"],
                      label="One photo's placement"),
               Line2D([], [], color=FUSED, lw=2, ls="--",
                      label="Fused estimate")]
    if (~a).any():
        handles.append(Line2D([], [], marker="x", ls="", color=INK2,
                              mew=2, label="Rejected (disagrees with the rest)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9)

    for col, k, unit, ylabel, title in (
            (0, 0, 100, "Position error (cm)", "Cube position error over the run"),
            (1, 1, 1, "Orientation error (deg)", "Cube orientation error over the run")):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(steps[seen], ew[seen, k] * unit, "o", color=CUBE_C["floor"], ms=3.5,
                alpha=0.7, mec="none", label="One photo")
        ax.plot(steps, ef[:, k] * unit, color=FUSED, lw=2,
                label="Fused so far")
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlim(0, len(T) + 1)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                  fontsize=9)
    fig.suptitle("Is the cube's estimated location consistent as the robot moves?",
                 x=0.01, ha="left", color=INK, fontweight="bold")
    fig.savefig(os.path.join(OUT, "cube_consistency.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    example_frame()
    print("D. Single photos ...", flush=True)
    Dr = experiment_D()
    print(json.dumps({k: v for k, v in Dr.items() if k not in ("by_faces_seen",
                                                               "by_distance")},
                     indent=1, default=float), flush=True)
    print("D2. Variations ...", flush=True)
    D2 = experiment_D2()
    print("E. Moving robot ...", flush=True)
    Er = experiment_E()
    print(json.dumps(Er["detail_run"], indent=1, default=float))
    with open(os.path.join(OUT, "cube_results.json"), "w") as f:
        json.dump({"D_single_photos": Dr, "D2_variations": D2,
                   "E_moving_robot": Er}, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}/")
