"""
tagsim.py - synthetic test bench for single-camera AprilTag localization
of a small floor robot.

What it does
------------
1. Builds a virtual arena (walls with AprilTags at known world poses).
2. Renders the image a robot-mounted camera would see from a known robot
   pose (perspective warp of each tag, anti-aliasing, blur, noise, JPEG),
   including the camera bobbing a legged robot produces while walking.
3. Runs a real detector (OpenCV ArUco, AprilTag 36h11 dictionary) on the
   rendered image, exactly as you would on the real robot.
4. Recovers the robot's global pose (x, y, yaw) four ways:
     single : one tag, IPPE_SQUARE PnP (closest / largest tag)
     multi  : all visible tag corners in one joint 6-DOF PnP
     floor_rigid : all corners, only x, y, yaw free; camera assumed to sit
                   exactly at its nominal height and angle (3-DOF)
     floor       : as above, but body roll / pitch / height wobble are
                   estimated too, with priors from how much the body bobs
5. Compares the estimates against ground truth.
6. Optionally places a cube with a tag on every face somewhere in the
   arena and estimates its pose relative to the robot two ways:
     pnp   : 6-DOF PnP over all visible cube-tag corners (IPPE for one face)
     floor : cube assumed resting on the floor (one face down), so only its
             x, y, yaw are free; camera wobble estimated as for the robot
   Combined with the robot's own pose estimate this places the cube on the
   map; CubeTracker fuses those placements over many photos.

Conventions
-----------
World : x, y on the floor, z up (metres).
Robot : x forward, y left, z up.  Pose (x, y, yaw).
Camera: OpenCV - x right, y down, z forward.
Tag   : x right, y up, z out of the tag towards the viewer;
        corners in ArUco order TL, TR, BR, BL.
Cube  : origin at the cube centre; tag k sits on face k (see CUBE_FACES).

Everything here is pure geometry + OpenCV, so the detection/pose code can
be copied to the real robot unchanged; only render() is simulation.
"""

from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy.optimize import least_squares

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


# ----------------------------------------------------------------------------
# Small SE(3) helpers
# ----------------------------------------------------------------------------
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# Camera axes expressed in the robot frame (camera looks along robot +x).
R_ROBOT_CAM_LEVEL = np.array([[0, 0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]], dtype=float)


# ----------------------------------------------------------------------------
# Scene description
# ----------------------------------------------------------------------------
@dataclass
class Tag:
    id: int
    center: np.ndarray      # world position of the tag centre (m)
    normal: np.ndarray      # unit vector pointing out of the tag into the room
    size: float             # side of the black square (m)
    R: np.ndarray = None    # full orientation (tag axes as columns); default
                            # is upright on a wall, tag x horizontal
    quiet: float = 1.5      # side of the white square around it / size
    shade: float = 1.0      # brightness factor when rendered

    @property
    def T_world_tag(self):
        if self.R is not None:
            return make_T(self.R, self.center)
        z = self.normal / np.linalg.norm(self.normal)
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return make_T(np.column_stack([x, y, z]), self.center)

    def corners_tag(self, scale=1.0):
        h = self.size * scale / 2
        return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)

    def corners_world(self, scale=1.0):
        T = self.T_world_tag
        return (T[:3, :3] @ self.corners_tag(scale).T).T + T[:3, 3]


def make_arena(width=4.0, depth=3.0, tag_size=0.15, tag_height=0.20,
               spacing=0.75):
    """Rectangular arena with tags along all four walls, facing inwards."""
    tags, tid = [], 0
    walls = [  # (start, end, inward normal)
        (np.array([0, 0]), np.array([width, 0]), np.array([0, 1])),
        (np.array([width, 0]), np.array([width, depth]), np.array([-1, 0])),
        (np.array([width, depth]), np.array([0, depth]), np.array([0, -1])),
        (np.array([0, depth]), np.array([0, 0]), np.array([1, 0])),
    ]
    for a, b, n in walls:
        L = np.linalg.norm(b - a)
        k = max(1, int(round(L / spacing)))
        for i in range(k):
            p = a + (b - a) * (i + 0.5) / k
            tags.append(Tag(tid, np.array([p[0], p[1], tag_height]),
                            np.array([n[0], n[1], 0.0]), tag_size))
            tid += 1
    return tags, (width, depth)


# Tag frame on each cube face, in cube coordinates: (outward normal, tag up).
CUBE_FACES = [((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)),
              ((-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, 1)),
              ((0, 0, 1), (-1, 0, 0)), ((0, 0, -1), (1, 0, 0))]
LIGHT = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


@dataclass
class Cube:
    """A cube with a tag centred on each face.  T is the true pose of the
    cube in the world; the robot only knows the geometry (model())."""
    T: np.ndarray = field(default_factory=lambda: np.eye(4))
    edge: float = 0.10      # side of the cube (m)
    tag_size: float = 0.07  # side of the black square on each face (m)
    first_id: int = 100

    @property
    def ids(self):
        return set(range(self.first_id, self.first_id + 6))

    def tags(self, T=None):
        """The six face tags, placed by T (default: the true pose)."""
        T = self.T if T is None else T
        out = []
        for k, (n, up) in enumerate(CUBE_FACES):
            z, y = np.array(n, float), np.array(up, float)
            R = T[:3, :3] @ np.column_stack([np.cross(y, z), y, z])
            c = T[:3, :3] @ (z * self.edge / 2) + T[:3, 3]
            shade = 0.72 + 0.28 * max(0.0, R[:, 2] @ LIGHT)
            out.append(Tag(self.first_id + k, c, R[:, 2], self.tag_size, R,
                           quiet=self.edge / self.tag_size, shade=shade))
        return out

    def model(self):
        """The tags in cube coordinates - what the robot knows."""
        return self.tags(np.eye(4))

    def corners(self, T=None):
        """The 8 cube corners, placed by T (default: the true pose)."""
        T = self.T if T is None else T
        h = self.edge / 2
        v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
        return (T[:3, :3] @ v.T).T + T[:3, 3]


def rest_rotation(axis, sign):
    """Rotation that turns cube axis `axis` (0, 1, 2), signed, to point up."""
    return {(2, 1): np.eye(3), (2, -1): rot_x(np.pi),
            (0, 1): rot_y(-np.pi / 2), (0, -1): rot_y(np.pi / 2),
            (1, 1): rot_x(np.pi / 2), (1, -1): rot_x(-np.pi / 2)}[(axis, sign)]


def snap_to_rest(R):
    """Nearest orientation of a cube lying flat on the floor:
    returns (R_rest, yaw) with R ~ rot_z(yaw) @ R_rest."""
    up = R.T @ np.array([0.0, 0.0, 1.0])      # world up in cube coordinates
    axis = int(np.argmax(np.abs(up)))
    R_rest = rest_rotation(axis, int(np.sign(up[axis])))
    M = R @ R_rest.T
    return R_rest, np.arctan2(M[1, 0], M[0, 0])


def random_cube_pose(rng, xlim, ylim, edge):
    """Cube lying on a random face with random yaw somewhere in the box."""
    R = rot_z(rng.uniform(-np.pi, np.pi)) @ rest_rotation(
        int(rng.integers(3)), int(rng.choice([-1, 1])))
    return make_T(R, [rng.uniform(*xlim), rng.uniform(*ylim), edge / 2])


def rot_err_deg(Ra, Rb):
    """Angle of the rotation between two orientations (deg)."""
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


@dataclass
class Camera:
    width: int = 640
    height: int = 480
    hfov_deg: float = 70.0
    mount_xyz: tuple = (0.05, 0.0, 0.10)   # camera position on robot (m)
    mount_pitch_deg: float = 0.0           # + = tilted up

    @property
    def f(self):
        return (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)

    @property
    def K(self):
        return np.array([[self.f, 0, (self.width - 1) / 2],
                         [0, self.f, (self.height - 1) / 2],
                         [0, 0, 1]])

    def T_robot_cam(self, roll=0.0, pitch=0.0, dz=0.0):
        """Camera pose on the robot; roll/pitch/dz model body bobbing."""
        R_body = rot_x(roll) @ rot_y(-pitch - np.radians(self.mount_pitch_deg))
        t = np.array(self.mount_xyz) + np.array([0, 0, dz])
        return make_T(R_body @ R_ROBOT_CAM_LEVEL, t)


def frame_coverage(points, cam, T_world_cam):
    """How much of an object (given by its world points, e.g. cube corners)
    the camera has in its frame: "full", "partial" or "none".
    Pure geometry with the true camera - says nothing about detection."""
    T_cam_world = inv_T(T_world_cam)
    pc = (T_cam_world[:3, :3] @ points.T).T + T_cam_world[:3, 3]
    front = pc[:, 2] > 1e-6
    uv = (cam.K @ pc[front].T).T
    uv = uv[:, :2] / uv[:, 2:]
    inside = np.zeros(len(points), bool)
    inside[front] = ((uv[:, 0] >= 0) & (uv[:, 0] <= cam.width - 1) &
                     (uv[:, 1] >= 0) & (uv[:, 1] <= cam.height - 1))
    return "full" if inside.all() else "partial" if inside.any() else "none"


def robot_T(x, y, yaw):
    return make_T(rot_z(yaw), [x, y, 0.0])


def pose_from_T(T):
    return T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])


# ----------------------------------------------------------------------------
# Rendering (simulation only)
# ----------------------------------------------------------------------------
@dataclass
class Degradation:
    blur_sigma: float = 0.7      # optics / focus blur (px)
    motion_px: float = 0.0       # length of linear motion blur (px)
    noise_sigma: float = 3.0     # sensor noise (grey levels)
    jpeg_quality: int = 80
    bob_pitch_deg: float = 2.0   # std of body pitch while walking
    bob_roll_deg: float = 1.5
    bob_dz_m: float = 0.005


_TAG_TEX = {}


def tag_texture(tid, px=320, margin=0.25):
    """Tag image with a white quiet zone; returns image and the quiet-zone scale."""
    if (tid, margin) not in _TAG_TEX:
        inner = cv2.aruco.generateImageMarker(ARUCO_DICT, tid, px)
        m = int(px * margin)
        tex = cv2.copyMakeBorder(inner, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
        _TAG_TEX[tid, margin] = tex
    tex = _TAG_TEX[tid, margin]
    return tex, tex.shape[0] / (tex.shape[0] - 2 * int(px * margin))


def _background(h, w, rng):
    """Low-frequency grey texture: lighter walls above, darker floor below."""
    base = rng.normal(0, 1, (h // 16 + 2, w // 16 + 2)).astype(np.float32)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    img = 140 + 18 * base
    img[h // 2:, :] -= 45
    return img


def render(tags, cam, T_world_cam, deg: Degradation, rng, ss=2):
    """Render the 8-bit grey image the camera sees (supersampled ss x).
    Each tag is drawn as a flat square of side size * quiet (tag plus white
    border); cube faces are just tags whose border reaches the cube edges.
    Painter's algorithm: fine for wall tags plus one convex cube."""
    W, H = cam.width * ss, cam.height * ss
    K = cam.K.copy()
    K[0, 0] *= ss
    K[1, 1] *= ss
    K[0, 2] = cam.K[0, 2] * ss + (ss - 1) / 2
    K[1, 2] = cam.K[1, 2] * ss + (ss - 1) / 2

    img = _background(H, W, rng)
    T_cam_world = inv_T(T_world_cam)
    cam_pos = T_world_cam[:3, 3]

    visible = []
    for tag in tags:
        tex, qz = tag_texture(tag.id, margin=(tag.quiet - 1) / 2)
        cw = tag.corners_world(qz)
        pc = (T_cam_world[:3, :3] @ cw.T).T + T_cam_world[:3, 3]
        if np.any(pc[:, 2] < 0.05):
            continue
        if np.dot(tag.normal, cam_pos - tag.center) <= 0:
            continue                                # seen from behind
        uv = (K @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:]
        if (uv[:, 0].max() < 0 or uv[:, 0].min() > W or
                uv[:, 1].max() < 0 or uv[:, 1].min() > H):
            continue
        visible.append((pc[:, 2].mean(), tex, uv, tag.shade))

    for _, tex, uv, shade in sorted(visible, key=lambda v: -v[0]):  # far to near
        s = tex.shape[0] - 1
        src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
        Hm = cv2.getPerspectiveTransform(src, np.float32(uv))
        warped = cv2.warpPerspective(tex.astype(np.float32), Hm, (W, H),
                                     flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.ones_like(tex, np.float32), Hm, (W, H),
                                   flags=cv2.INTER_LINEAR)
        # Mild shading so tags are not perfectly black/white.
        warped = (25 + warped * (200 / 255)) * shade
        img = img * (1 - mask) + warped * mask

    img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
    if deg.blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), deg.blur_sigma)
    if deg.motion_px > 0.5:
        L = int(np.ceil(deg.motion_px))
        k = np.zeros((2 * L + 1, 2 * L + 1), np.float32)
        ang = rng.uniform(0, np.pi)
        c = L
        for t in np.linspace(-deg.motion_px / 2, deg.motion_px / 2, 4 * L + 1):
            k[int(round(c + t * np.sin(ang))), int(round(c + t * np.cos(ang)))] = 1
        img = cv2.filter2D(img, -1, k / k.sum())
    img = img + rng.normal(0, deg.noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if deg.jpeg_quality < 100:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, deg.jpeg_quality])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


# ----------------------------------------------------------------------------
# Detection + pose estimation (this part runs unchanged on the real robot)
# ----------------------------------------------------------------------------
def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 3
    return cv2.aruco.ArucoDetector(ARUCO_DICT, p)


def detect(detector, img, tag_map):
    corners, ids, _ = detector.detectMarkers(img)
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        if int(i) in tag_map:                    # ignore unknown IDs
            out.append((tag_map[int(i)], c.reshape(4, 2).astype(np.float64)))
    return out


def estimate_single(det, K, dist, T_robot_cam):
    """Pose from the largest visible tag (IPPE_SQUARE, best of 2 solutions)."""
    tag, uv = max(det, key=lambda d: cv2.contourArea(d[1].astype(np.float32)))
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        tag.corners_tag(), uv, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    R, _ = cv2.Rodrigues(rvecs[0])
    T_cam_tag = make_T(R, tvecs[0])
    T_world_cam = tag.T_world_tag @ inv_T(T_cam_tag)
    return T_world_cam @ inv_T(T_robot_cam), tag


def estimate_multi(det, K, dist, T_robot_cam):
    """Joint 6-DOF PnP over every corner of every visible tag."""
    if len(det) == 1:
        return estimate_single(det, K, dist, T_robot_cam)[0]
    obj = np.vstack([t.corners_world() for t, _ in det])
    img = np.vstack([uv for _, uv in det])
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T_world_cam = inv_T(make_T(R, tvec))
    return T_world_cam @ inv_T(T_robot_cam)


def estimate_floor(det, K, dist, cam, init_xyyaw, rigid=False,
                   prior_roll_deg=3.0, prior_pitch_deg=3.0, prior_dz_m=0.01,
                   pixel_sigma=0.5):
    """Floor-constrained estimate: the robot stands on the floor and the
    camera mount is known from the robot's design.

    rigid=True : only (x, y, yaw) are free - assumes the camera never tilts.
    rigid=False: (x, y, yaw) free, plus body roll / pitch / height wobble
                 with Gaussian priors around the nominal mount.  This is a
                 small MAP problem: corner reprojection error (px) + priors.
    Robust (Huber) loss on the corner residuals.
    """
    obj = np.vstack([t.corners_world() for t, _ in det])
    img = np.vstack([uv for _, uv in det])
    priors = np.array([np.radians(prior_roll_deg), np.radians(prior_pitch_deg),
                       prior_dz_m])

    def resid(p):
        wob = np.zeros(3) if rigid else p[3:]
        T_world_cam = robot_T(*p[:3]) @ cam.T_robot_cam(*wob)
        T_cam_world = inv_T(T_world_cam)
        rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
        proj, _ = cv2.projectPoints(obj, rvec, T_cam_world[:3, 3], K, dist)
        r = (proj.reshape(-1, 2) - img).ravel() / pixel_sigma
        if rigid:
            return r
        return np.concatenate([r, wob / priors])

    x0 = np.asarray(init_xyyaw, float)
    if not rigid:
        x0 = np.concatenate([x0, np.zeros(3)])
    r = least_squares(resid, x0, loss="huber", f_scale=3.0)
    return r.x[:3]


def estimate_object_pnp(det, K, dist):
    """Candidate poses T_cam_obj of a rigid multi-tag object, best first.

    det holds (tag in object coordinates, corners px).  With one face in
    view the pose is ambiguous (a flat square looks almost the same tilted
    either way), so both IPPE solutions are returned.  With two or more
    faces at an angle the corners are not coplanar and SQPNP + LM gives a
    single answer.
    """
    if len(det) == 1:
        tag, uv = det[0]
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            tag.corners_tag(), uv, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return [make_T(cv2.Rodrigues(rv)[0], tv) @ inv_T(tag.T_world_tag)
                for rv, tv in zip(rvecs, tvecs)]
    obj = np.vstack([t.corners_world() for t, _ in det])
    img = np.vstack([uv for _, uv in det])
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    return [make_T(cv2.Rodrigues(rvec)[0], tvec)]


def estimate_object_floor(det, K, dist, cam, candidates, z_center,
                          prior_roll_deg=3.0, prior_pitch_deg=3.0,
                          prior_dz_m=0.01, pixel_sigma=0.5, return_wobble=False):
    """Pose T_robot_obj of a cube lying flat on the floor (one face down,
    centre at height z_center), seen by the robot's camera.

    Free: the cube's x, y, yaw on the floor (robot frame) plus the camera's
    roll / pitch / height wobble with priors, as in estimate_floor().
    Each candidate T_cam_obj (e.g. both IPPE solutions) is snapped to the
    nearest lying-flat orientation and refined; the lowest cost wins, which
    usually resolves the one-face ambiguity.
    return_wobble: also return the estimated camera (roll, pitch, dz).
    """
    obj = np.vstack([t.corners_world() for t, _ in det])
    img = np.vstack([uv for _, uv in det])
    priors = np.array([np.radians(prior_roll_deg), np.radians(prior_pitch_deg),
                       prior_dz_m])
    T_rc = cam.T_robot_cam()

    def pose(p, R_rest):
        return make_T(rot_z(p[2]) @ R_rest, [p[0], p[1], z_center])

    best = None
    for T_co in candidates:
        T_ro = T_rc @ T_co
        R_rest, yaw = snap_to_rest(T_ro[:3, :3])

        def resid(p):
            T_cam_obj = inv_T(cam.T_robot_cam(*p[3:])) @ pose(p, R_rest)
            rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
            proj, _ = cv2.projectPoints(obj, rvec, T_cam_obj[:3, 3], K, dist)
            r = (proj.reshape(-1, 2) - img).ravel() / pixel_sigma
            return np.concatenate([r, p[3:] / priors])

        x0 = np.array([T_ro[0, 3], T_ro[1, 3], yaw, 0, 0, 0])
        r = least_squares(resid, x0, loss="huber", f_scale=3.0)
        if best is None or r.cost < best[0]:
            best = (r.cost, pose(r.x, R_rest), r.x[3:])
    return best[1:] if return_wobble else best[1]


class CubeTracker:
    """Fuses per-photo cube placements (world frame) into one estimate.

    Position: weighted mean, weight 1 / sd^2 with sd growing with distance.
    Orientation: weighted chordal mean of the rotation matrices.
    Once a few photos agree, a placement that disagrees by more than
    gate_m or gate_deg (e.g. a flipped one-face solution) is rejected.
    """

    def __init__(self, gate_m=0.15, gate_deg=20.0, min_votes=3):
        self.gate_m, self.gate_deg, self.min_votes = gate_m, gate_deg, min_votes
        self.w, self.p, self.R = [], [], []

    @property
    def T(self):
        if not self.w:
            return None
        w = np.array(self.w)
        p = (w[:, None] * np.array(self.p)).sum(0) / w.sum()
        U, _, Vt = np.linalg.svd((w[:, None, None] * np.array(self.R)).sum(0))
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        return make_T(R, p)

    def add(self, T_world_obj, dist):
        """Returns True if the placement was accepted."""
        if len(self.w) >= self.min_votes:
            T = self.T
            if (np.linalg.norm(T_world_obj[:2, 3] - T[:2, 3]) > self.gate_m or
                    rot_err_deg(T_world_obj[:3, :3], T[:3, :3]) > self.gate_deg):
                return False
        sd = 0.01 + 0.02 * dist
        self.w.append(1 / sd ** 2)
        self.p.append(T_world_obj[:3, 3].copy())
        self.R.append(T_world_obj[:3, :3].copy())
        return True


# ----------------------------------------------------------------------------
# One full measurement: render from a true pose, detect, estimate
# ----------------------------------------------------------------------------
@dataclass
class Result:
    n_tags: int
    nearest_dist: float = np.nan
    tag_px: float = np.nan
    err: dict = field(default_factory=dict)   # method -> (pos_err_m, yaw_err_deg)
    est: dict = field(default_factory=dict)   # method -> (x, y, yaw)
    # Cube (only when measure() is given one)
    cube_in_frame: str = ""                   # "full" / "partial" / "none"
    cube_faces: int = 0                       # cube faces detected
    cube_dist: float = np.nan                 # camera to cube centre (m)
    cube_px: float = np.nan                   # largest cube tag side (px)
    cube_rel: dict = field(default_factory=dict)      # method -> T_robot_cube
    cube_err_rel: dict = field(default_factory=dict)  # method -> (xy_m, rot_deg)
    cube_err_world: dict = field(default_factory=dict)  # same, via robot fix


def cube_error(T_est, T_true):
    """(horizontal position error m, rotation error deg)."""
    return (np.linalg.norm(T_est[:2, 3] - T_true[:2, 3]),
            rot_err_deg(T_est[:3, :3], T_true[:3, :3]))


def perturb_map(tags, rng, pos_sigma=0.01, yaw_sigma_deg=1.0):
    """Model tape-measure errors: the map the robot uses differs from
    where the tags really are by pos_sigma (m) and yaw_sigma_deg."""
    out = []
    for t in tags:
        dp = rng.normal(0, pos_sigma, 3)
        dp[2] *= 0.5
        n = rot_z(np.radians(rng.normal(0, yaw_sigma_deg))) @ t.normal
        out.append(Tag(t.id, t.center + dp, n, t.size))
    return out


def perturb_intrinsics(K, rng=None, f_err=0.03, c_err_px=8.0):
    """A miscalibrated K: focal length off by f_err (fraction), principal
    point off by c_err_px in a random direction."""
    rng = rng or np.random.default_rng(0)
    K2 = K.copy()
    K2[0, 0] *= 1 + f_err
    K2[1, 1] *= 1 + f_err
    a = rng.uniform(0, 2 * np.pi)
    K2[0, 2] += c_err_px * np.cos(a)
    K2[1, 2] += c_err_px * np.sin(a)
    return K2


def measure(true_pose, tags, cam, deg, rng, detector=None,
            K_est=None, dist_est=None, map_tags=None, return_image=False,
            cube=None, cube_floor=True):
    """tags     : where the tags really are (used for rendering)
    map_tags : where the robot believes they are (default: the same)
    cube     : optional Cube placed in the scene; its pose is estimated
               relative to the robot ("pnp", and "floor" if cube_floor) and,
               when wall tags give a robot fix, in the world."""
    detector = detector or make_detector()
    tag_map = {t.id: t for t in (map_tags or tags)}
    if cube is not None:
        tag_map.update({t.id: t for t in cube.model()})
        tags = tags + cube.tags()

    # True camera pose includes random body bobbing.
    roll = np.radians(rng.normal(0, deg.bob_roll_deg))
    pitch = np.radians(rng.normal(0, deg.bob_pitch_deg))
    dz = rng.normal(0, deg.bob_dz_m)
    T_world_cam_true = robot_T(*true_pose) @ cam.T_robot_cam(roll, pitch, dz)
    img = render(tags, cam, T_world_cam_true, deg, rng)

    # The estimator only knows the nominal (non-bobbing) mount.
    T_rc = cam.T_robot_cam()
    K = cam.K if K_est is None else K_est
    dist = np.zeros(5) if dist_est is None else dist_est

    det = detect(detector, img, tag_map)
    cube_ids = cube.ids if cube is not None else set()
    cdet = [d for d in det if d[0].id in cube_ids]
    det = [d for d in det if d[0].id not in cube_ids]
    res = Result(n_tags=len(det))
    if cube is not None:
        res.cube_in_frame = frame_coverage(cube.corners(), cam, T_world_cam_true)
    if det:
        cam_pos = T_world_cam_true[:3, 3]
        res.nearest_dist = min(np.linalg.norm(t.center - cam_pos) for t, _ in det)
        res.tag_px = max(np.sqrt(cv2.contourArea(uv.astype(np.float32)))
                         for _, uv in det)

        T_s, _ = estimate_single(det, K, dist, T_rc)
        T_m = estimate_multi(det, K, dist, T_rc)
        p_s, p_m = pose_from_T(T_s), pose_from_T(T_m)
        p_r = estimate_floor(det, K, dist, cam, p_m, rigid=True)
        p_f = estimate_floor(det, K, dist, cam, p_m)
        for name, p in (("single", p_s), ("multi", p_m),
                        ("floor_rigid", p_r), ("floor", p_f)):
            e_pos = np.hypot(p[0] - true_pose[0], p[1] - true_pose[1])
            e_yaw = abs(np.degrees(wrap(p[2] - true_pose[2])))
            res.err[name] = (e_pos, e_yaw)
            res.est[name] = tuple(p)

    if cdet:
        res.cube_faces = len(cdet)
        res.cube_dist = np.linalg.norm(cube.T[:3, 3] - T_world_cam_true[:3, 3])
        res.cube_px = max(np.sqrt(cv2.contourArea(uv.astype(np.float32)))
                          for _, uv in cdet)
        cands = estimate_object_pnp(cdet, K, dist)
        res.cube_rel["pnp"] = T_rc @ cands[0]
        if cube_floor:
            res.cube_rel["floor"] = estimate_object_floor(
                cdet, K, dist, cam, cands, cube.edge / 2)
        T_rel_true = inv_T(robot_T(*true_pose)) @ cube.T
        for m, T in res.cube_rel.items():
            res.cube_err_rel[m] = cube_error(T, T_rel_true)
            if det:
                res.cube_err_world[m] = cube_error(
                    robot_T(*res.est["floor"]) @ T, cube.T)
    if return_image:
        return res, img, det + cdet
    return res
