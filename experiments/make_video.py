"""
make_video.py - videos of one simulated run (the experiment C trajectory)

  python3 -m experiments.make_video [out_dir] [--steps 240] [--substeps 3] [--hold 2] [--fps 24]
  (from the repo root; default output: results/videos)

Writes three H.264 MP4s (needs the ffmpeg command-line tool):
  camera_view.mp4      what the robot's camera sees, with detections
  trajectory_anim.mp4  animated map + position-error plot
  combined.mp4         both side by side (1600 x 900)

Each step of the run becomes:
  --substeps walking frames : the camera moves between poses (motion blur,
                              body bobbing); estimates show the odometry
                              prediction only
  --hold photo frames       : the robot stops, takes the photo the
                              simulation actually used, tags are detected
                              and the fused estimate snaps to the fix
The run itself is identical to experiment C: rendering the in-between
frames uses its own random generator.
"""
import sys, os, argparse, subprocess, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from tagsim import render, robot_T, wrap, Degradation
from experiments.run_experiments import (simulate_run, plain_log, INK, INK2, MUTED, GRID,
                             AXIS, SURF, C)

BLUE, ORANGE = C["single"], C["multi"]           # fused estimate, dead reckoning


def bgr(hex_):
    h = hex_.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


# ---------------------------------------------------------------------------
# Video output through ffmpeg
# ---------------------------------------------------------------------------
class VideoWriter:
    def __init__(self, path, w, h, fps):
        assert w % 2 == 0 and h % 2 == 0, "H.264 needs even dimensions"
        self.path, self.w, self.h = path, w, h
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        assert frame.shape == (self.h, self.w, 3), frame.shape
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


# ---------------------------------------------------------------------------
# Camera view
# ---------------------------------------------------------------------------
def annotate_camera(gray, dets, cam, hud=None):
    """Grey camera image -> BGR with tag outlines, IDs, axes and a status bar."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for tag, uv in dets:
        cv2.polylines(vis, [np.round(uv).astype(np.int32)], True, bgr(BLUE), 2,
                      cv2.LINE_AA)
        ok, rv, tv = cv2.solvePnP(tag.corners_tag(), uv, cam.K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            cv2.drawFrameAxes(vis, cam.K, None, rv, tv, tag.size * 0.6, 2)
        x0, y0 = uv[:, 0].min(), uv[:, 1].min()
        cv2.putText(vis, f"id {tag.id}", (int(x0), int(y0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr(BLUE), 1, cv2.LINE_AA)
    if hud:
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], 26), (20, 20, 20), -1)
        vis = cv2.addWeighted(overlay, 0.65, vis, 0.35, 0)
        cv2.putText(vis, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 239, 236), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Animated figure: camera (left), map + error (right)
# ---------------------------------------------------------------------------
FIG_W, FIG_H, SPLIT_X = 1600, 900, 760           # px; right panel = x >= SPLIT_X
ROBOT_SHAPE = np.array([[0.14, 0.0], [-0.08, 0.07], [-0.08, -0.07]])


def robot_poly(pose, scale=1.0):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    return (R @ (scale * ROBOT_SHAPE).T).T + pose[:2]


def fov_poly(pose, hfov_deg, rng_m=1.4):
    a = np.radians(hfov_deg) / 2
    angs = np.linspace(pose[2] - a, pose[2] + a, 12)
    pts = [pose[:2]] + [pose[:2] + rng_m * np.array([np.cos(t), np.sin(t)])
                        for t in angs]
    return np.array(pts)


class Figure:
    def __init__(self, ctx):
        self.ctx = ctx
        W, D, steps = ctx["W"], ctx["D"], ctx["steps"]
        self.xlim, self.ylim = (-0.6, W + 0.6), (-0.6, D + 0.6)
        plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
        self.fig = fig = plt.figure(figsize=(FIG_W / 100, FIG_H / 100), dpi=100,
                                    facecolor=SURF)
        px = lambda x, y, w, h: [x / FIG_W, 1 - (y + h) / FIG_H, w / FIG_W, h / FIG_H]

        # Left: camera.  720 x 540 px axes for a 640 x 480 image.
        fig.text(20 / FIG_W, 1 - 40 / FIG_H, "Robot camera (simulated, 640 x 480)",
                 color=INK, fontsize=14, fontweight="bold", va="bottom")
        self.ax_cam = fig.add_axes(px(20, 52, 720, 540))
        self.ax_cam.set_axis_off()
        self.im = self.ax_cam.imshow(np.zeros((480, 640, 3), np.uint8),
                                     interpolation="bilinear")
        self.status_title = fig.text(20 / FIG_W, 1 - 625 / FIG_H, "", color=INK,
                                     fontsize=14, fontweight="bold")
        self.status = fig.text(20 / FIG_W, 1 - 650 / FIG_H, "", color=INK2,
                               fontsize=11, family="DejaVu Sans Mono", va="top",
                               linespacing=1.6)

        # Right top: map.
        ax = self.ax_map = fig.add_axes(px(SPLIT_X + 70, 52, 740, 430))
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_xlabel("x (m)", color=INK2, fontsize=10)
        ax.set_ylabel("y (m)", color=INK2, fontsize=10)
        ax.set_title("Where the robot thinks it is", loc="left", color=INK,
                     fontsize=14, fontweight="bold", pad=10)
        ax.add_patch(Rectangle((0, 0), W, D, fill=False, ec=AXIS, lw=1.5))
        tc = np.array([t.center[:2] for t in ctx["tags"]])
        ax.plot(tc[:, 0], tc[:, 1], "s", color=MUTED, ms=5, zorder=2)
        self.seen = ax.plot([], [], "s", color=BLUE, ms=8, zorder=3)[0]
        self.sight = LineCollection([], colors=BLUE, linewidths=1,
                                    alpha=0.6, zorder=2)
        ax.add_collection(self.sight)
        self.fov = Polygon(np.zeros((3, 2)), closed=True, fc=INK, alpha=0.07,
                           ec="none", zorder=1)
        ax.add_patch(self.fov)
        self.wp = ax.plot([], [], "x", color=MUTED, ms=9, mew=2, zorder=2)[0]
        self.l_true = ax.plot([], [], color=INK, lw=2, zorder=4)[0]
        self.l_dr = ax.plot([], [], color=ORANGE, lw=2, zorder=4)[0]
        self.l_kf = ax.plot([], [], color=BLUE, lw=2, ls="--", zorder=5)[0]
        # Estimates are drawn as larger outlines so the true robot (solid,
        # on top) stays visible inside them when they agree.
        self.r_dr = Polygon(np.zeros((3, 2)), fc="none", ec=ORANGE, lw=2, zorder=6)
        self.r_kf = Polygon(np.zeros((3, 2)), fc="none", ec=BLUE, lw=2, zorder=7)
        self.r_true = Polygon(np.zeros((3, 2)), fc=INK, ec=SURF, lw=1, zorder=8)
        for p in (self.r_true, self.r_dr, self.r_kf):
            ax.add_patch(p)
        self.off_map = ax.text(0, 0, "", color=INK2, fontsize=9, zorder=8)
        handles = [Line2D([], [], color=INK, lw=2, label="True path"),
                   Line2D([], [], color=BLUE, lw=2, ls="--",
                          label="Dead reckoning + tag fixes"),
                   Line2D([], [], color=ORANGE, lw=2, label="Motor dead reckoning"),
                   Line2D([], [], marker="s", ls="", color=MUTED, ms=6, label="Tag"),
                   Line2D([], [], marker="s", ls="", color=BLUE, ms=7,
                          label="Tag detected in photo"),
                   Line2D([], [], marker="x", ls="", color=MUTED, ms=8, mew=2,
                          label="Next waypoint")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                  ncol=3, fontsize=9, frameon=False, labelcolor=INK2)

        # Right bottom: error over time.
        ax = self.ax_err = fig.add_axes(px(SPLIT_X + 70, 655, 740, 185))
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.set_yscale("log")
        ax.set_xlim(0, steps * 1.1 + 2)
        ax.set_ylim(0.3, 1000)
        plain_log(ax)
        ax.set_xlabel("Step", color=INK2, fontsize=10)
        ax.set_ylabel("Position error (cm)", color=INK2, fontsize=10)
        ax.set_title("Position error vs truth", loc="left", color=INK,
                     fontsize=14, fontweight="bold", pad=10)
        self.e_dr = ax.plot([], [], color=ORANGE, lw=2,
                            label="Motor dead reckoning")[0]
        self.e_kf = ax.plot([], [], color=BLUE, lw=2,
                            label="Dead reckoning + tag fixes")[0]
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                  fontsize=9, frameon=False, labelcolor=INK2)
        self.t_dr = ax.text(0, 1, "", color=INK2, fontsize=9, va="center")
        self.t_kf = ax.text(0, 1, "", color=INK2, fontsize=9, va="center")

    def draw(self, cam_bgr, st):
        """st: dict with current poses, path histories, errors, status."""
        self.im.set_data(cam_bgr[:, :, ::-1])
        self.status_title.set_text(st["title"])
        self.status.set_text(st["status"])

        pose = st["true"]
        self.fov.set_xy(fov_poly(pose, self.ctx["cam"].hfov_deg))
        self.wp.set_data([st["waypoint"][0]], [st["waypoint"][1]])
        for line, key in ((self.l_true, "true"), (self.l_dr, "dr"), (self.l_kf, "kf")):
            h = st["paths"][key]
            line.set_data(h[:, 0], h[:, 1])
        self.r_true.set_xy(robot_poly(st["true"]))
        self.r_kf.set_xy(robot_poly(st["kf"], 1.6))

        # Dead-reckoning marker is pinned to the edge once it leaves the map.
        dr = st["dr"].copy()
        clamped = dr.copy()
        clamped[0] = np.clip(dr[0], self.xlim[0] + 0.12, self.xlim[1] - 0.12)
        clamped[1] = np.clip(dr[1], self.ylim[0] + 0.12, self.ylim[1] - 0.12)
        self.r_dr.set_xy(robot_poly(clamped, 1.6))
        if np.allclose(clamped[:2], dr[:2]):
            self.off_map.set_text("")
        else:
            self.off_map.set_text("dead reckoning\noff the map")
            ha = "right" if clamped[0] > np.mean(self.xlim) else "left"
            va = "top" if clamped[1] > np.mean(self.ylim) else "bottom"
            dx = -0.18 if ha == "right" else 0.18
            dy = -0.12 if va == "top" else 0.12
            self.off_map.set_position((clamped[0] + dx, clamped[1] + dy))
            self.off_map.set_ha(ha)
            self.off_map.set_va(va)

        seen = st["seen_tags"]
        if seen:
            c = np.array([t.center[:2] for t in seen])
            self.seen.set_data(c[:, 0], c[:, 1])
            self.sight.set_segments([[pose[:2], p] for p in c])
        else:
            self.seen.set_data([], [])
            self.sight.set_segments([])

        e = st["errors"]
        if len(e):
            ks = np.arange(1, len(e) + 1)
            self.e_dr.set_data(ks, e[:, 0])
            self.e_kf.set_data(ks, e[:, 1])
            n = len(e) + 2
            # End-of-line labels, pushed apart (in log space) if they collide.
            y_dr, y_kf = np.log10(np.clip(e[-1], 0.35, 900))
            gap = 0.28
            if abs(y_dr - y_kf) < gap:
                mid = (y_dr + y_kf) / 2
                hi, lo = mid + gap / 2, mid - gap / 2
                y_dr, y_kf = (hi, lo) if y_dr >= y_kf else (lo, hi)
            self.t_dr.set_position((n, 10 ** y_dr))
            self.t_dr.set_text(f"{e[-1, 0]:.0f} cm")
            self.t_kf.set_position((n, 10 ** y_kf))
            self.t_kf.set_text(f"{e[-1, 1]:.1f} cm")

        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, 2::-1])   # RGBA -> BGR


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def interp(a, b, t):
    out = a + (b - a) * t
    out[2] = wrap(a[2] + wrap(b[2] - a[2]) * t)
    return out


def fmt_pose(name, p, truth=None):
    s = f"{name:<16s} x {p[0]:5.2f} m   y {p[1]:5.2f} m   heading {np.degrees(p[2]):6.1f} deg"
    if truth is not None:
        s += f"   err {100 * np.hypot(p[0] - truth[0], p[1] - truth[1]):5.1f} cm"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="results/videos")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--substeps", type=int, default=3)
    ap.add_argument("--hold", type=int, default=2)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--preview", type=int, nargs="*", default=[],
                    help="also save PNGs of these combined-video frame numbers")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Run the simulation once without recording to get the scene, then
    # replay it with the recorder (same seed -> same run).
    _, ctx = simulate_run(1)
    ctx["steps"] = a.steps
    fig = Figure(ctx)
    cam = ctx["cam"]
    rng_vis = np.random.default_rng(99)
    walk_deg = Degradation(motion_px=4.0)          # bobbing + blur while walking

    w_cam = VideoWriter(os.path.join(a.out, "camera_view.mp4"), 640, 480, a.fps)
    w_traj = VideoWriter(os.path.join(a.out, "trajectory_anim.mp4"),
                         FIG_W - SPLIT_X, FIG_H, a.fps)
    w_comb = VideoWriter(os.path.join(a.out, "combined.mp4"), FIG_W, FIG_H, a.fps)
    paths = {k: [ctx["start"].copy()] for k in ("true", "dr", "kf")}
    errors = []
    frame_no = [0]
    t0 = time.time()

    def emit(cam_gray, dets, poses, title, status, hud, waypoint, seen):
        cam_bgr = annotate_camera(cam_gray, dets, cam)
        st = {"true": poses["true"], "dr": poses["dr"], "kf": poses["kf"],
              "paths": {k: np.array(paths[k] + [poses[k]]) for k in paths},
              "errors": np.array(errors), "title": title, "status": status,
              "waypoint": waypoint, "seen_tags": seen}
        frame = fig.draw(cam_bgr, st)
        w_comb.write(frame)
        w_traj.write(frame[:, SPLIT_X:])
        w_cam.write(annotate_camera(cam_gray, dets, cam, hud=hud))
        if frame_no[0] in a.preview:
            cv2.imwrite(os.path.join(a.out, f"preview_{frame_no[0]:05d}.png"), frame)
        frame_no[0] += 1

    def on_step(info):
        k, prev, new = info["k"], info["prev"], info["new"]
        fwd, turn = info["cmd"]
        action = (f"walk {100 * fwd:.0f} cm" if fwd > 0 else "turn on the spot")
        action += f", turn {np.degrees(turn):+.0f} deg"
        step_txt = f"Step {k + 1} / {a.steps}"

        # Walking frames: camera between poses, estimates = odometry only.
        for i in range(1, a.substeps + 1):
            t = i / (a.substeps + 1)
            poses = {"true": interp(prev["true"], new["true"], t),
                     "dr": interp(prev["dr"], new["dr"], t),
                     "kf": interp(prev["kf"], info["kf_pred"], t)}
            roll = np.radians(rng_vis.normal(0, walk_deg.bob_roll_deg))
            pitch = np.radians(rng_vis.normal(0, walk_deg.bob_pitch_deg))
            dz = rng_vis.normal(0, walk_deg.bob_dz_m)
            T_wc = robot_T(*poses["true"]) @ cam.T_robot_cam(roll, pitch, dz)
            img = render(ctx["tags"], cam, T_wc, walk_deg, rng_vis)
            status = "\n".join([f"Command: {action}",
                                fmt_pose("True pose", poses["true"]),
                                fmt_pose("Fused estimate", poses["kf"], poses["true"]),
                                fmt_pose("Dead reckoning", poses["dr"], poses["true"]),
                                "",
                                "Walking: the estimate is moved by the motor",
                                "command alone until the next photo."])
            emit(img, [], poses, f"{step_txt}  -  walking", status,
                 f"STEP {k + 1}/{a.steps}  WALKING", info["waypoint"], [])

        # Photo frames: the image the simulation used; estimate snaps to fix.
        for key in paths:
            paths[key].append(new[key].copy())
        errors.append([100 * np.hypot(*(new["dr"] - new["true"])[:2]),
                       100 * np.hypot(*(new["kf"] - new["true"])[:2])])
        r = info["result"]
        n = r.n_tags
        if n and info["fix"]:
            verdict = f"{n} tag{'s' * (n > 1)} detected - fix accepted"
        elif n:
            verdict = f"{n} tag{'s' * (n > 1)} detected - fix rejected (outlier)"
        else:
            verdict = "no tags visible - keep dead reckoning"
        lines = [f"Command: {action}",
                 fmt_pose("True pose", new["true"]),
                 fmt_pose("Fused estimate", new["kf"], new["true"]),
                 fmt_pose("Dead reckoning", new["dr"], new["true"]), ""]
        if n:
            f = r.est["floor"]
            lines += [fmt_pose("Tag measurement", np.array(f), new["true"]),
                      f"{'':<16s} nearest tag {r.nearest_dist:.2f} m, "
                      f"largest {r.tag_px:.0f} px"]
        seen = [t for t, _ in info["dets"]]
        for _ in range(a.hold):
            emit(info["image"], info["dets"], new,
                 f"{step_txt}  -  photo: {verdict}", "\n".join(lines),
                 f"STEP {k + 1}/{a.steps}  PHOTO  {verdict.upper()}",
                 info["waypoint"], seen)
        if (k + 1) % 20 == 0:
            print(f"  step {k + 1}/{a.steps}  ({frame_no[0]} frames, "
                  f"{time.time() - t0:.0f} s)", flush=True)

    log, _ = simulate_run(a.steps, on_step=on_step)
    for w in (w_cam, w_traj, w_comb):
        w.close()
    secs = frame_no[0] / a.fps
    print(f"wrote {frame_no[0]} frames ({secs:.0f} s at {a.fps} fps) to "
          f"{a.out}/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
