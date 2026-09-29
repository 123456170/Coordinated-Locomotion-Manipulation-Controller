import streamlit as st
import numpy as np
import cv2
import math
import os
import time
import tempfile
from typing import Dict, Optional, Tuple

st.set_page_config(
    page_title="Whole-Body Gesture Humanoid Controller",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------------
try:
    import osqp
    from scipy import sparse
    OSQP_OK = True
except Exception:
    OSQP_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False

try:
    import pybullet as p
    PYBULLET_OK = True
except Exception:
    PYBULLET_OK = False

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
GROUND_Z = 0.0
TARGET_Z = 0.95

THIGH = 0.42
SHANK = 0.40
FOOT_LEN = 0.18
TORSO = 0.52
UPPER_ARM = 0.33
FOREARM = 0.31
HEAD_R = 0.09
PELVIS_Z = 0.82

VAR_NAMES = ["torso", "hip_l", "knee_l", "hip_r", "knee_r", "shoulder", "elbow", "z"]

VAR_LOWER = np.array([-0.45, -1.10, -0.05, -1.10, -0.05, -2.80, -2.60, 0.62])
VAR_UPPER = np.array([0.75, 1.15, 1.75, 1.15, 1.75, 1.40, 0.35, 0.95])

BG = (18, 22, 30)
PANEL = (28, 34, 46)
PANEL_BORDER = (70, 80, 100)
TEXT = (230, 235, 245)
DIM_TEXT = (170, 180, 195)
GOOD = (80, 220, 120)
WARN = (80, 200, 255)
BAD = (80, 120, 255)
ACCENT = (255, 200, 0)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))


def lerp(a, b, t):
    return a + (b - a) * clamp(t, 0.0, 1.0)


def acos_clamp(x):
    return math.acos(clamp(x, -1.0, 1.0))


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------
# Kinematics and metrics
# ------------------------------------------------------------------
def forward_kinematics(pose: Dict) -> Dict:
    x = pose.get("x", 0.0)
    z = pose.get("z", PELVIS_Z)

    pelvis = np.array([x, z], dtype=float)
    torso_ang = pose.get("torso", 0.05)

    shoulder = pelvis + np.array([math.sin(torso_ang) * TORSO, math.cos(torso_ang) * TORSO])
    head = shoulder + np.array([math.sin(torso_ang) * 0.18, math.cos(torso_ang) * 0.18])

    sh = pose.get("shoulder", -1.25)
    el = pose.get("elbow", -0.45)

    elbow = shoulder + np.array([math.cos(sh) * UPPER_ARM, math.sin(sh) * UPPER_ARM])
    wrist = elbow + np.array([math.cos(sh + el) * FOREARM, math.sin(sh + el) * FOREARM])

    def leg_fk(hip, knee):
        thigh_ang = -math.pi / 2.0 + hip
        knee_pos = pelvis + np.array([math.cos(thigh_ang) * THIGH, math.sin(thigh_ang) * THIGH])
        shank_ang = thigh_ang - knee
        ankle_pos = knee_pos + np.array([math.cos(shank_ang) * SHANK, math.sin(shank_ang) * SHANK])
        foot_pos = ankle_pos + np.array([FOOT_LEN * 0.8, 0.0])
        return knee_pos, ankle_pos, foot_pos

    knee_l, ankle_l, foot_l = leg_fk(pose.get("hip_l", 0.0), pose.get("knee_l", 0.1))
    knee_r, ankle_r, foot_r = leg_fk(pose.get("hip_r", 0.0), pose.get("knee_r", 0.1))

    return {
        "pelvis": pelvis,
        "shoulder": shoulder,
        "head": head,
        "elbow": elbow,
        "wrist": wrist,
        "knee_l": knee_l,
        "ankle_l": ankle_l,
        "foot_l": foot_l,
        "knee_r": knee_r,
        "ankle_r": ankle_r,
        "foot_r": foot_r,
    }


def compute_metrics(pose: Dict, fk: Dict, phase: float) -> Dict:
    pts = np.array([
        fk["pelvis"],
        fk["shoulder"],
        fk["head"],
        fk["elbow"],
        fk["wrist"],
        fk["knee_l"],
        fk["ankle_l"],
        fk["knee_r"],
        fk["ankle_r"],
    ], dtype=float)

    weights = np.array([0.42, 0.22, 0.10, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    com = np.average(pts, axis=0, weights=weights)

    ph = phase % 0.5
    double = ph < 0.09 or ph > 0.41

    foot_l = fk["foot_l"]
    foot_r = fk["foot_r"]

    if double:
        smin = min(foot_l[0], foot_r[0]) - FOOT_LEN * 0.40
        smax = max(foot_l[0], foot_r[0]) + FOOT_LEN * 0.80
    else:
        stance = foot_r if phase < 0.5 else foot_l
        smin = stance[0] - FOOT_LEN * 0.50
        smax = stance[0] + FOOT_LEN * 0.90

    margin = min(com[0] - smin, smax - com[0])
    phase_name = "Double Support" if double else ("Left Swing" if phase < 0.5 else "Right Swing")

    return {
        "com": com,
        "support": (smin, smax),
        "margin": float(margin),
        "double": bool(double),
        "phase_name": phase_name,
    }


# ------------------------------------------------------------------
# Controllers
# ------------------------------------------------------------------
def gait_command(phase: float, speed: float, x: float) -> Dict:
    amp = clamp(0.22 + 2.4 * speed, 0.18, 0.65)
    knee_amp = clamp(0.30 + 2.0 * speed, 0.22, 0.85)

    hip_l = amp * math.sin(2.0 * math.pi * phase)
    hip_r = amp * math.sin(2.0 * math.pi * (phase + 0.5))

    knee_l = knee_amp * max(0.0, math.sin(2.0 * math.pi * phase + 0.45)) + 0.08
    knee_r = knee_amp * max(0.0, math.sin(2.0 * math.pi * (phase + 0.5) + 0.45)) + 0.08

    torso = 0.05 + 0.25 * speed + 0.025 * math.sin(4.0 * math.pi * phase)
    pelvis_z = PELVIS_Z + 0.018 * math.sin(4.0 * math.pi * phase)

    ankle_l = 0.08 * math.sin(2.0 * math.pi * phase + 1.1)
    ankle_r = 0.08 * math.sin(2.0 * math.pi * (phase + 0.5) + 1.1)

    return {
        "x": x,
        "z": pelvis_z,
        "torso": torso,
        "hip_l": hip_l,
        "knee_l": knee_l,
        "hip_r": hip_r,
        "knee_r": knee_r,
        "ankle_l": ankle_l,
        "ankle_r": ankle_r,
        "shoulder": -1.25 + 0.08 * math.sin(2.0 * math.pi * phase),
        "elbow": -0.45,
    }


def arm_ik(pose: Dict, target: np.ndarray) -> Dict:
    fk = forward_kinematics(pose)
    shoulder = fk["shoulder"]

    dx = target[0] - shoulder[0]
    dz = target[1] - shoulder[1]

    d = clamp(math.hypot(dx, dz), 0.08, UPPER_ARM + FOREARM - 0.03)
    base = math.atan2(dz, dx)

    alpha = acos_clamp((UPPER_ARM * UPPER_ARM + d * d - FOREARM * FOREARM) / (2.0 * UPPER_ARM * d))
    elbow_inner = acos_clamp((UPPER_ARM * UPPER_ARM + FOREARM * FOREARM - d * d) / (2.0 * UPPER_ARM * FOREARM))

    shoulder_abs = base + alpha
    elbow_rel = -(math.pi - elbow_inner)

    shoulder_abs = clamp(shoulder_abs, VAR_LOWER[5], VAR_UPPER[5])
    elbow_rel = clamp(elbow_rel, VAR_LOWER[6], VAR_UPPER[6])

    return {"shoulder": shoulder_abs, "elbow": elbow_rel}


def get_vars(pose: Dict) -> np.ndarray:
    return np.array([pose.get(k, 0.0) for k in VAR_NAMES], dtype=float)


def set_vars(pose: Dict, vars: np.ndarray) -> Dict:
    out = pose.copy()
    for i, k in enumerate(VAR_NAMES):
        out[k] = float(vars[i])
    return out


def com_x_for_vars(base_pose: Dict, vars: np.ndarray, phase: float) -> float:
    pose = set_vars(base_pose, vars)
    pose["phase"] = phase
    fk = forward_kinematics(pose)
    metrics = compute_metrics(pose, fk, phase)
    return float(metrics["com"][0])


def solve_qp(H: np.ndarray, g: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    H = 0.5 * (H + H.T)
    n = H.shape[0]

    if OSQP_OK:
        try:
            A = sparse.csc_matrix(sparse.eye(n))
            solver = osqp.OSQP()
            solver.setup(
                P=sparse.csc_matrix(H),
                q=g,
                A=A,
                l=lower,
                u=upper,
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=400,
                polish=True,
            )
            res = solver.solve()
            if res.x is not None and np.all(np.isfinite(res.x)):
                return np.clip(res.x, lower, upper)
        except Exception:
            pass

    try:
        x = np.linalg.solve(H + 1e-6 * np.eye(n), -g)
    except Exception:
        x = np.linalg.lstsq(H + 1e-5 * np.eye(n), -g, rcond=None)[0]

    return np.clip(x, lower, upper)


def whole_body_mpc(
    current_pose: Dict,
    gait_pose: Dict,
    reach_pose: Dict,
    phase: float,
    metrics: Dict,
    reach_allowed: bool,
    reach_scale: float = 1.0,
) -> Dict:
    vars0 = get_vars(current_pose)

    gait_full = current_pose.copy()
    gait_full.update(gait_pose)
    gait_full["x"] = current_pose.get("x", 0.0)
    vars_gait = get_vars(gait_full)

    vars_reach = vars0.copy()
    vars_reach[5] = reach_pose.get("shoulder", vars0[5])
    vars_reach[6] = reach_pose.get("elbow", vars0[6])
    vars_reach[0] = clamp(vars_gait[0] + 0.10 * reach_scale, VAR_LOWER[0], VAR_UPPER[0])

    n = len(VAR_NAMES)
    H = 1e-6 * np.eye(n)
    g = np.zeros(n)

    w_gait = np.array([1.20, 2.50, 2.00, 2.50, 2.00, 0.15, 0.15, 2.00])
    w_reach = np.zeros(n)

    if reach_allowed:
        w_reach[5] = 3.00 * reach_scale
        w_reach[6] = 3.00 * reach_scale
        w_reach[0] = 0.35 * reach_scale
    else:
        w_reach[5] = 0.05
        w_reach[6] = 0.05

    w_smooth = np.full(n, 0.25)
    w_total = w_gait + w_reach + w_smooth

    ref = (w_gait * vars_gait + w_reach * vars_reach + w_smooth * vars0) / np.maximum(1e-8, w_total)

    H += np.diag(w_total)
    g += -w_total * ref

    margin = metrics["margin"]
    if margin < 0.0:
        stab_w = 30.0
    elif margin < 0.02:
        stab_w = 20.0
    elif margin < 0.05:
        stab_w = 8.0
    else:
        stab_w = 1.5

    eps = 1e-3
    com0 = com_x_for_vars(current_pose, vars0, phase)
    G = np.zeros(n)

    for j in range(n):
        v = vars0.copy()
        v[j] += eps
        G[j] = (com_x_for_vars(current_pose, v, phase) - com0) / eps

    support_min, support_max = metrics["support"]
    safety = 0.025

    if support_max - support_min < 2.0 * safety:
        desired = 0.5 * (support_min + support_max)
    else:
        desired = clamp(0.5 * (support_min + support_max), support_min + safety, support_max - safety)

    b = com0 - float(np.dot(G, vars0)) - desired

    H += stab_w * np.outer(G, G)
    g += stab_w * b * G

    x = solve_qp(H, g, VAR_LOWER, VAR_UPPER)
    new_pose = set_vars(current_pose, x)

    new_pose["x"] = current_pose.get("x", 0.0)
    new_pose["phase"] = phase
    new_pose["ankle_l"] = gait_pose.get("ankle_l", 0.0)
    new_pose["ankle_r"] = gait_pose.get("ankle_r", 0.0)

    return new_pose


def naive_combine(
    current_pose: Dict,
    gait_pose: Dict,
    reach_pose: Dict,
    reach_effort: float = 0.0,
    destabilize: bool = False,
) -> Dict:
    pose = current_pose.copy()
    pose.update(gait_pose)

    pose["shoulder"] = reach_pose.get("shoulder", pose.get("shoulder", -1.25))
    pose["elbow"] = reach_pose.get("elbow", pose.get("elbow", -0.45))

    if reach_effort > 0.1:
        pose["torso"] = pose.get("torso", 0.05) + 0.24 * reach_effort
        pose["hip_l"] = pose.get("hip_l", 0.0) + 0.06 * reach_effort
        pose["hip_r"] = pose.get("hip_r", 0.0) + 0.06 * reach_effort

    if destabilize:
        pose["torso"] = pose.get("torso", 0.05) + 0.10

    return pose


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
class HumanoidSim:
    def __init__(self, mode: str = "coordinated"):
        self.t = 0.0
        self.mode = mode
        self.phase = 0.0
        self.base_speed = 0.08
        self.speed = 0.08
        self.object_x = 1.25
        self.goal_x = 2.0
        self.force_reach = False
        self.reach_active = False
        self.reach_queued = False
        self.reach_progress = 0.0
        self.com_margin = 0.1
        self.phase_name = "Double Support"
        self.status = "Ready"
        self.fall_timer = 0.0
        self.trail = []

        self.pose = {
            "x": 0.0,
            "z": PELVIS_Z,
            "torso": 0.05,
            "hip_l": 0.0,
            "knee_l": 0.1,
            "hip_r": 0.0,
            "knee_r": 0.1,
            "ankle_l": 0.0,
            "ankle_r": 0.0,
            "shoulder": -1.25,
            "elbow": -0.45,
            "fall": 0.0,
            "phase": 0.0,
        }

        fk = forward_kinematics(self.pose)
        self.metrics = compute_metrics(self.pose, fk, self.phase)
        self.last_arm_target = fk["wrist"].copy()
        self.current_target = fk["wrist"].copy()

    def configure_segment(self, cfg: Dict):
        self.mode = cfg.get("mode", "coordinated")
        self.pose["x"] = cfg.get("x", 0.0)
        self.pose["z"] = PELVIS_Z
        self.pose["fall"] = 0.0

        self.phase = cfg.get("phase", 0.0)
        self.pose["phase"] = self.phase

        self.base_speed = cfg.get("speed", 0.08)
        self.speed = self.base_speed

        self.object_x = cfg.get("object_x", 1.25)
        self.goal_x = cfg.get("goal_x", max(2.0, self.object_x + 0.5))

        self.force_reach = cfg.get("reach_active", False)
        self.reach_active = self.force_reach
        self.reach_queued = False
        self.fall_timer = 0.0
        self.trail = []
        self.status = "Segment ready"

        gait = gait_command(self.phase, self.speed, self.pose["x"])
        for k in ["z", "torso", "hip_l", "knee_l", "hip_r", "knee_r", "ankle_l", "ankle_r", "shoulder", "elbow"]:
            if k in gait:
                self.pose[k] = gait[k]

        fk = forward_kinematics(self.pose)
        self.metrics = compute_metrics(self.pose, fk, self.phase)
        self.com_margin = self.metrics["margin"]
        self.phase_name = self.metrics["phase_name"]
        self.last_arm_target = fk["wrist"].copy()
        self.current_target = fk["wrist"].copy()

    def gesture_target(self) -> np.ndarray:
        if self.reach_active or self.force_reach or self.pose["x"] > 0.35:
            return np.array([self.object_x, TARGET_Z], dtype=float)
        return np.array([self.pose["x"] + 0.18, self.pose["z"] + 0.12], dtype=float)

    def is_double_support(self) -> bool:
        ph = self.phase % 0.5
        return ph < 0.09 or ph > 0.41

    def step(self, dt: float, mode: Optional[str] = None, force_queue: bool = False, destabilize: bool = False):
        mode = mode or self.mode
        self.mode = mode
        self.t += dt
        self.pose["phase"] = self.phase

        if self.pose.get("fall", 0.0) < 0.05:
            target_speed = self.base_speed

            if self.pose["x"] >= self.goal_x:
                target_speed = 0.0

            if self.reach_active and abs(self.object_x - self.pose["x"]) < 0.28:
                target_speed = min(target_speed, 0.022)

            self.speed = lerp(self.speed, target_speed, 0.12)
            self.pose["x"] += self.speed * dt
            self.phase = (self.phase + dt * (0.65 + 4.5 * self.speed)) % 1.0
        else:
            self.speed = max(0.0, self.speed - dt * 0.5)

        self.pose["phase"] = self.phase

        if self.force_reach or self.pose["x"] > 0.35:
            self.reach_active = True

        target = self.gesture_target()
        self.current_target = target

        fk_current = forward_kinematics(self.pose)
        metrics_current = compute_metrics(self.pose, fk_current, self.phase)

        reach_dist = float(np.linalg.norm(target - fk_current["shoulder"]))
        reach_effort = clamp((reach_dist - 0.20) / 0.60, 0.0, 1.0)

        double = metrics_current["double"]

        if mode == "coordinated":
            large = reach_effort > 0.42
            allowed = (not large) or double

            if force_queue and large:
                allowed = False

            if self.reach_active and not allowed:
                self.reach_queued = True
                arm_target = self.last_arm_target
                self.status = "Large reach queued - waiting for double support"
            else:
                self.reach_queued = False
                self.last_arm_target = target.copy()
                arm_target = target
                self.status = "Reach executing" if self.reach_active else "Walking"

            gait = gait_command(self.phase, self.speed, self.pose["x"])
            reach = arm_ik(self.pose, arm_target)
            reach_scale = 1.0 if self.reach_active else 0.25

            self.pose = whole_body_mpc(
                self.pose,
                gait,
                reach,
                self.phase,
                metrics_current,
                not self.reach_queued,
                reach_scale,
            )
        else:
            arm_target = target
            self.reach_queued = False
            self.status = "Naive: gait and arm independent"

            gait = gait_command(self.phase, self.speed, self.pose["x"])
            reach = arm_ik(self.pose, arm_target)

            self.pose = naive_combine(
                self.pose,
                gait,
                reach,
                reach_effort,
                destabilize,
            )

        fk = forward_kinematics(self.pose)
        metrics = compute_metrics(self.pose, fk, self.phase)

        self.metrics = metrics
        self.com_margin = metrics["margin"]
        self.phase_name = metrics["phase_name"]

        if mode == "naive" and (destabilize or (self.reach_active and metrics["margin"] < 0.012)):
            self.fall_timer += dt
        else:
            self.fall_timer = max(0.0, self.fall_timer - dt)

        if self.fall_timer > 0.35 or (mode == "naive" and destabilize and self.t > 1.0 and metrics["margin"] < 0.04):
            self.pose["fall"] = min(1.0, self.pose.get("fall", 0.0) + dt * 1.8)

        fall = self.pose.get("fall", 0.0)
        if fall > 0.0:
            self.pose["torso"] = clamp(self.pose.get("torso", 0.05) + dt * 2.2 * fall, -0.6, 1.7)
            self.pose["z"] = clamp(self.pose.get("z", PELVIS_Z) - dt * 0.8 * fall, 0.30, 1.0)
            self.pose["shoulder"] = clamp(self.pose.get("shoulder", -1.25) + dt * 2.0 * fall, VAR_LOWER[5], VAR_UPPER[5])
            self.pose["elbow"] = clamp(self.pose.get("elbow", -0.45) - dt * 1.0 * fall, VAR_LOWER[6], VAR_UPPER[6])
            self.pose["hip_l"] = clamp(self.pose.get("hip_l", 0.0) + dt * 1.5 * fall, VAR_LOWER[1], VAR_UPPER[1])
            self.pose["hip_r"] = clamp(self.pose.get("hip_r", 0.0) + dt * 1.4 * fall, VAR_LOWER[3], VAR_UPPER[3])
            self.pose["knee_l"] = clamp(self.pose.get("knee_l", 0.1) + dt * 1.2 * fall, VAR_LOWER[2], VAR_UPPER[2])
            self.pose["knee_r"] = clamp(self.pose.get("knee_r", 0.1) + dt * 1.2 * fall, VAR_LOWER[4], VAR_UPPER[4])

        fk2 = forward_kinematics(self.pose)
        wrist = fk2["wrist"]

        if self.reach_active:
            initial = max(0.35, abs(self.object_x - self.pose["x"]) + 0.35)
            d = float(np.linalg.norm(wrist - target))
            self.reach_progress = clamp(1.0 - d / initial, 0.0, 1.0)
        else:
            self.reach_progress = 0.0

        self.trail.append((self.pose["x"], self.pose["z"], wrist[0], wrist[1]))
        if len(self.trail) > 45:
            self.trail.pop(0)


# ------------------------------------------------------------------
# Storyboard segments
# ------------------------------------------------------------------
def segment_info(t: float) -> Dict:
    if t < 12.0:
        return {
            "name": "Coordinated Walk + Reach",
            "mode": "coordinated",
            "x": 0.0,
            "phase": 0.0,
            "speed": 0.085,
            "object_x": 1.25,
            "reach_active": False,
            "goal_x": 2.0,
        }

    if t < 27.0:
        return {
            "name": "Naive Baseline Destabilizes",
            "mode": "naive",
            "x": 0.82,
            "phase": 0.26,
            "speed": 0.015,
            "object_x": 1.38,
            "reach_active": True,
            "destabilize": True,
            "goal_x": 2.0,
        }

    if t < 40.0:
        return {
            "name": "Coordinated Recovery",
            "mode": "coordinated",
            "x": 0.80,
            "phase": 0.0,
            "speed": 0.035,
            "object_x": 1.25,
            "reach_active": True,
            "goal_x": 2.0,
        }

    if t < 52.0:
        return {
            "name": "Double-Support Timing Gate",
            "mode": "coordinated",
            "x": 0.92,
            "phase": 0.27,
            "speed": 0.020,
            "object_x": 1.58,
            "reach_active": True,
            "force_queue": t < 47.0,
            "goal_x": 2.0,
        }

    return {
        "name": "Benchmark Comparison",
        "mode": "coordinated",
        "x": 1.05,
        "phase": 0.0,
        "speed": 0.008,
        "object_x": 1.35,
        "reach_active": False,
        "chart_focus": True,
        "goal_x": 2.0,
    }


def create_sim_for_segment(seg: Dict) -> HumanoidSim:
    sim = HumanoidSim(mode=seg.get("mode", "coordinated"))
    sim.configure_segment(seg)
    return sim


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------
def world_to_px(x: float, z: float, cam_x: float, rect: Tuple[int, int, int, int], scale: int = 125):
    px = rect[0] + rect[2] // 2 + int((x - cam_x) * scale)
    py = rect[1] + rect[3] - 30 - int(z * scale)
    return (px, py)


def draw_panel(img: np.ndarray, rect: Tuple[int, int, int, int], title: str):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BORDER, 1)
    cv2.putText(img, title, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT, 1, cv2.LINE_AA)


def draw_humanoid(img: np.ndarray, fk: Dict, rect: Tuple[int, int, int, int], cam_x: float, color, style: str = "robot"):
    def pt(v):
        return world_to_px(v[0], v[1], cam_x, rect)

    pelvis = pt(fk["pelvis"])
    shoulder = pt(fk["shoulder"])
    head = pt(fk["head"])
    elbow = pt(fk["elbow"])
    wrist = pt(fk["wrist"])

    knee_l = pt(fk["knee_l"])
    ankle_l = pt(fk["ankle_l"])
    foot_l = pt(fk["foot_l"])

    knee_r = pt(fk["knee_r"])
    ankle_r = pt(fk["ankle_r"])
    foot_r = pt(fk["foot_r"])

    cv2.polylines(img, [np.array([pelvis, knee_l, ankle_l, foot_l], dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array([pelvis, knee_r, ankle_r, foot_r], dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    cv2.line(img, pelvis, shoulder, color, 3, cv2.LINE_AA)
    cv2.line(img, shoulder, elbow, color, 2, cv2.LINE_AA)
    cv2.line(img, elbow, wrist, color, 2, cv2.LINE_AA)

    head_r = max(4, int(HEAD_R * 125))
    cv2.circle(img, head, head_r, color, 2, cv2.LINE_AA)

    if style == "perception":
        joints = [pelvis, shoulder, elbow, wrist, knee_l, ankle_l, foot_l, knee_r, ankle_r, foot_r]
        for j in joints:
            cv2.circle(img, j, 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_scene_common(img: np.ndarray, sim: HumanoidSim, fk: Dict, rect: Tuple[int, int, int, int], kind: str):
    cam_x = sim.pose["x"]

    gy = rect[1] + rect[3] - 30
    cv2.line(img, (rect[0] + 5, gy), (rect[0] + rect[2] - 5, gy), (60, 65, 80), 2)

    obj_base = world_to_px(sim.object_x, 0.0, cam_x, rect)
    obj_top = world_to_px(sim.object_x, TARGET_Z, cam_x, rect)

    cv2.line(img, obj_base, obj_top, (100, 100, 100), 3, cv2.LINE_AA)
    cv2.circle(img, obj_top, 7, (0, 170, 255), -1, cv2.LINE_AA)

    trail = sim.trail
    n = len(trail)
    for i, item in enumerate(trail):
        a = (i + 1) / max(1, n)
        p = world_to_px(item[2], item[3], cam_x, rect)
        col = (int(255 * a * 0.55), int(200 * a * 0.55), 0)
        cv2.circle(img, p, 2, col, -1, cv2.LINE_AA)

    if kind == "physics":
        smin, smax = sim.metrics["support"]
        p1 = world_to_px(smin, 0.0, cam_x, rect)
        p2 = world_to_px(smax, 0.0, cam_x, rect)
        cv2.line(img, p1, p2, GOOD, 4, cv2.LINE_AA)

        com_pt = world_to_px(sim.metrics["com"][0], sim.metrics["com"][1], cam_x, rect)

        if sim.com_margin > 0.02:
            com_color = GOOD
        elif sim.com_margin > 0.0:
            com_color = WARN
        else:
            com_color = BAD

        cv2.circle(img, com_pt, 6, com_color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"CoM margin {sim.com_margin * 100:.1f} cm",
            (rect[0] + 10, rect[1] + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            TEXT,
            1,
            cv2.LINE_AA,
        )

    if sim.mode == "coordinated":
        body_color = GOOD if kind == "physics" else (255, 190, 0)
    else:
        body_color = BAD if kind == "physics" else (255, 190, 0)

    draw_humanoid(img, fk, rect, cam_x, body_color, style=kind)

    if sim.reach_active:
        wrist_pt = world_to_px(fk["wrist"][0], fk["wrist"][1], cam_x, rect)
        target_pt = world_to_px(sim.current_target[0], sim.current_target[1], cam_x, rect)
        cv2.line(img, wrist_pt, target_pt, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(img, target_pt, 6, (0, 170, 255), 2, cv2.LINE_AA)

    if sim.reach_queued and kind == "physics":
        cv2.putText(
            img,
            "REACH QUEUED",
            (rect[0] + rect[2] - 135, rect[1] + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WARN,
            1,
            cv2.LINE_AA,
        )


def draw_phase_gauge(img: np.ndarray, cx: int, cy: int, r: int, phase: float, phase_name: str):
    cv2.circle(img, (cx, cy), r, (45, 50, 65), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (120, 130, 150), 2, cv2.LINE_AA)

    ang = 2.0 * math.pi * phase - math.pi / 2.0
    end = (int(cx + 0.78 * r * math.cos(ang)), int(cy + 0.78 * r * math.sin(ang)))
    cv2.line(img, (cx, cy), end, ACCENT, 3, cv2.LINE_AA)

    cv2.putText(img, "Gait", (cx - 16, cy - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, DIM_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, phase_name, (cx - 52, cy + r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT, 1, cv2.LINE_AA)


def draw_bar(img: np.ndarray, x: int, y: int, w: int, h: int, value: float, color, label: str):
    cv2.putText(img, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 50, 65), -1)

    fill_w = int(w * clamp(value, 0.0, 1.0))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)

    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 130, 150), 1)


def draw_chart(img: np.ndarray, rect: Tuple[int, int, int, int], logs: Optional[Dict], title: str = "CoM margin: naive vs coordinated"):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (25, 30, 42), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 90, 110), 1)

    cv2.putText(img, title, (x + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT, 1, cv2.LINE_AA)

    if not logs:
        return

    n = min(len(logs.get("t", [])), 240)
    if n < 2:
        return

    y_min, y_max = -0.15, 0.30
    plot_x = x + 38
    plot_y = y + 30
    plot_w = w - 48
    plot_h = h - 52

    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (40, 46, 62), -1)

    def val_to_px(i, val, length):
        val = clamp(val, y_min, y_max)
        px = plot_x + int((i / max(1, length - 1)) * plot_w)
        py = plot_y + plot_h - int((val - y_min) / (y_max - y_min) * plot_h)
        return (px, py)

    zero_y = plot_y + plot_h - int((0.0 - y_min) / (y_max - y_min) * plot_h)
    cv2.line(img, (plot_x, zero_y), (plot_x + plot_w, zero_y), (90, 95, 110), 1)

    series = [
        ("naive", (80, 120, 255)),
        ("coord", (80, 220, 120)),
    ]

    for key, color in series:
        vals = np.asarray(logs.get(key, [])[-n:], dtype=float)
        if len(vals) < 2:
            continue

        pts = []
        for i, v in enumerate(vals):
            pts.append(val_to_px(i, v, len(vals)))

        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    cv2.line(img, (x + 10, y + h - 14), (x + 30, y + h - 14), (80, 120, 255), 2)
    cv2.putText(img, "Naive", (x + 35, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, DIM_TEXT, 1, cv2.LINE_AA)

    cv2.line(img, (x + 105, y + h - 14), (x + 125, y + h - 14), (80, 220, 120), 2)
    cv2.putText(img, "Coordinated", (x + 130, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, DIM_TEXT, 1, cv2.LINE_AA)


def render_chart_image(logs: Dict, width: int = 430, height: int = 250) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = BG
    draw_chart(img, (0, 0, width, height), logs, title="CoM margin benchmark")
    return img


def render_frame(
    sim: HumanoidSim,
    logs: Optional[Dict],
    t: float,
    seg: Dict,
    width: int = 860,
    height: int = 380,
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = BG

    mode = seg.get("mode", sim.mode)
    name = seg.get("name", "Live demo")

    cv2.putText(
        img,
        "Whole-Body Gesture Humanoid Controller",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"t={t:5.1f}s | {name}",
        (14, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        DIM_TEXT,
        1,
        cv2.LINE_AA,
    )

    mode_color = GOOD if mode == "coordinated" else BAD
    cv2.putText(
        img,
        f"MODE: {mode.upper()}",
        (width - 220, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        mode_color,
        1,
        cv2.LINE_AA,
    )

    panel_h = height - 170
    left = (12, 52, width // 2 - 18, panel_h)
    right = (width // 2 + 6, 52, width // 2 - 18, panel_h)

    draw_panel(img, left, "Perception: gesture stream")
    draw_panel(img, right, "Physics twin: whole-body MPC")

    fk = forward_kinematics(sim.pose)

    draw_scene_common(img, sim, fk, left, "perception")
    draw_scene_common(img, sim, fk, right, "physics")

    draw_phase_gauge(img, 58, height - 68, 28, sim.phase, sim.phase_name)

    margin_norm = clamp((sim.com_margin + 0.10) / 0.35, 0.0, 1.0)
    if sim.com_margin > 0.02:
        margin_color = GOOD
    elif sim.com_margin > 0.0:
        margin_color = WARN
    else:
        margin_color = BAD

    draw_bar(img, 120, height - 98, 185, 15, sim.reach_progress, (0, 200, 255), "Reach task progress")
    draw_bar(
        img,
        120,
        height - 60,
        185,
        15,
        margin_norm,
        margin_color,
        f"CoM margin {sim.com_margin * 100:.1f} cm",
    )

    cv2.putText(
        img,
        sim.status,
        (330, height - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        TEXT,
        1,
        cv2.LINE_AA,
    )

    if logs is not None:
        draw_chart(img, (width - 350, height - 115, 340, 105), logs)

    if seg.get("chart_focus", False):
        draw_chart(img, (width // 4, 70, width // 2, height - 150), logs, title="Benchmark: CoM margin")

    return img


# ------------------------------------------------------------------
# Video generation
# ------------------------------------------------------------------
def open_video_writer(base_path: str, fps: int, size: Tuple[int, int]):
    mp4_path = base_path + ".mp4"
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, mp4_path, "video/mp4"

    avi_path = base_path + ".avi"
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if writer.isOpened():
        return writer, avi_path, "video/avi"

    return None, None, None


def generate_scripted_video(
    duration: float = 60.0,
    fps: int = 10,
    width: int = 860,
    height: int = 380,
    logs: Optional[Dict] = None,
    progress_cb=None,
    frame_cb=None,
) -> Tuple[bytes, str]:
    tmpdir = tempfile.mkdtemp()
    base = os.path.join(tmpdir, "demo")

    writer, path, mime = open_video_writer(base, fps, (width, height))
    if writer is None:
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass
        return b"", "video/mp4"

    sim = None
    current_name = None
    frames = int(duration * fps)
    dt = 1.0 / fps

    for i in range(frames):
        t = i * dt
        seg = segment_info(t)

        if sim is None or seg["name"] != current_name:
            sim = create_sim_for_segment(seg)
            current_name = seg["name"]

        sim.step(
            dt,
            mode=seg.get("mode", "coordinated"),
            force_queue=seg.get("force_queue", False),
            destabilize=seg.get("destabilize", False),
        )

        frame = render_frame(sim, logs, t, seg, width, height)
        writer.write(frame)

        if frame_cb is not None and i % 4 == 0:
            frame_cb(frame)

        if progress_cb is not None and i % 10 == 0:
            progress_cb(i / max(1, frames))

    writer.release()

    with open(path, "rb") as f:
        data = f.read()

    try:
        os.unlink(path)
        os.rmdir(tmpdir)
    except Exception:
        pass

    return data, mime


def generate_user_video(
    duration: float = 60.0,
    fps: int = 10,
    width: int = 800,
    height: int = 380,
    mode: str = "coordinated",
    speed: float = 0.08,
    object_x: float = 1.25,
    force_queue: bool = False,
    destabilize: bool = False,
    logs: Optional[Dict] = None,
    progress_cb=None,
    frame_cb=None,
) -> Tuple[bytes, str]:
    tmpdir = tempfile.mkdtemp()
    base = os.path.join(tmpdir, "user_demo")

    writer, path, mime = open_video_writer(base, fps, (width, height))
    if writer is None:
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass
        return b"", "video/mp4"

    sim = HumanoidSim(mode=mode)
    sim.configure_segment({
        "mode": mode,
        "x": 0.0,
        "phase": 0.0,
        "speed": speed,
        "object_x": object_x,
        "reach_active": True,
        "goal_x": max(2.0, object_x + 0.6),
    })

    frames = int(duration * fps)
    dt = 1.0 / fps

    seg = {
        "name": f"Interactive {mode.title()} Demo",
        "mode": mode,
    }

    for i in range(frames):
        t = i * dt

        sim.step(
            dt,
            mode=mode,
            force_queue=force_queue,
            destabilize=destabilize,
        )

        frame = render_frame(sim, logs, t, seg, width, height)
        writer.write(frame)

        if frame_cb is not None and i % 4 == 0:
            frame_cb(frame)

        if progress_cb is not None and i % 10 == 0:
            progress_cb(i / max(1, frames))

    writer.release()

    with open(path, "rb") as f:
        data = f.read()

    try:
        os.unlink(path)
        os.rmdir(tmpdir)
    except Exception:
        pass

    return data, mime


# ------------------------------------------------------------------
# Cached assets
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Building benchmark logs...")
def get_benchmark_logs(duration: float = 60.0, sample_fps: int = 5) -> Dict:
    logs = {"t": [], "naive": [], "coord": []}

    sim_c = None
    sim_n = None
    current_name = None

    dt = 1.0 / sample_fps
    frames = int(duration * sample_fps)

    for i in range(frames):
        t = i * dt
        seg = segment_info(t)

        if sim_c is None or sim_n is None or seg["name"] != current_name:
            cfg_c = dict(seg)
            cfg_c["mode"] = "coordinated"

            cfg_n = dict(seg)
            cfg_n["mode"] = "naive"

            sim_c = create_sim_for_segment(cfg_c)
            sim_n = create_sim_for_segment(cfg_n)

            current_name = seg["name"]

        sim_c.step(
            dt,
            mode="coordinated",
            force_queue=seg.get("force_queue", False),
            destabilize=False,
        )

        sim_n.step(
            dt,
            mode="naive",
            force_queue=False,
            destabilize=seg.get("destabilize", False),
        )

        logs["t"].append(t)
        logs["naive"].append(sim_n.com_margin)
        logs["coord"].append(sim_c.com_margin)

    return logs


@st.cache_data(show_spinner="Building startup demo video...")
def get_preview_video() -> Tuple[bytes, str]:
    logs = get_benchmark_logs(60.0, 5)
    return generate_scripted_video(
        duration=16.0,
        fps=10,
        width=760,
        height=360,
        logs=logs,
        progress_cb=None,
        frame_cb=None,
    )


# ------------------------------------------------------------------
# App UI
# ------------------------------------------------------------------
st.title("Coordinated Locomotion + Manipulation Controller")
st.caption(
    "Dual-controller supervisor: gait CPG + arm IK share a whole-body MPC/QP horizon. "
    "Balance wins automatically when CoM margin becomes small, and large reaches are gated through double support."
)

backend_text = (
    f"OSQP: {'ON' if OSQP_OK else 'fallback QP'} | "
    f"MediaPipe: {'ON' if MEDIAPIPE_OK else 'not available'} | "
    f"PyBullet: {'ON' if PYBULLET_OK else 'not available'}"
)
st.caption(backend_text)

logs = get_benchmark_logs(60.0, 5)

with st.sidebar:
    st.header("Controller settings")

    mode = st.radio(
        "Control mode",
        options=["coordinated", "naive"],
        index=0,
        horizontal=True,
    )

    speed = st.slider("Walking speed (m/s)", 0.02, 0.18, 0.085, 0.005)
    object_x = st.slider("Object position x (m)", 0.6, 1.9, 1.25, 0.05)
    record_duration = st.slider("Record duration (s)", 5, 60, 60, 5)

    force_queue = st.checkbox("Force double-support queue", value=False)
    destabilize = st.checkbox("Apply naive perturbation", value=True)

    st.divider()
    st.markdown(
        """
        **Demo behavior**
        - Coordinated mode uses shared MPC/QP.
        - Naive mode runs gait and arm independently.
        - Large reaches wait for double support in coordinated mode.
        - Balance weight increases when CoM margin drops below 2 cm.
        """
    )

top_col1, top_col2 = st.columns([3, 2])

with top_col1:
    st.subheader("Auto-loaded simulation demo")

    try:
        preview_bytes, preview_mime = get_preview_video()

        if preview_bytes:
            st.video(preview_bytes, format=preview_mime)

            ext = "mp4" if "mp4" in preview_mime else "avi"
            st.download_button(
                "Download startup demo",
                data=preview_bytes,
                file_name=f"wholebody_demo_preview.{ext}",
                mime=preview_mime,
            )
        else:
            st.warning("Video encoder is not available on this machine.")
    except Exception as e:
        st.warning(f"Startup video generation failed: {e}")

with top_col2:
    st.subheader("Benchmark CoM margin")
    chart_img = render_chart_image(logs, width=430, height=250)
    st.image(bgr_to_rgb(chart_img))

st.divider()

st.subheader("Interactive live controller")

live_box = st.empty()
progress_box = st.empty()

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    if st.button("Run 5 s live preview"):
        sim = HumanoidSim(mode=mode)
        sim.configure_segment({
            "mode": mode,
            "x": 0.0,
            "phase": 0.0,
            "speed": speed,
            "object_x": object_x,
            "reach_active": True,
            "goal_x": max(2.0, object_x + 0.6),
        })

        fps = 12
        dt = 1.0 / fps

        seg = {
            "name": "Interactive live preview",
            "mode": mode,
        }

        for i in range(int(5.0 * fps)):
            sim.step(
                dt,
                mode=mode,
                force_queue=force_queue,
                destabilize=(destabilize and mode == "naive"),
            )

            frame = render_frame(sim, logs, sim.t, seg, width=760, height=360)
            live_box.image(bgr_to_rgb(frame))
            time.sleep(dt)

with button_col2:
    if st.button("Record selected-duration demo"):
        progress = progress_box.progress(0)

        def progress_cb(p):
            progress.progress(int(100 * clamp(p, 0.0, 1.0)))

        def frame_cb(frame):
            live_box.image(bgr_to_rgb(frame))

        video_bytes, video_mime = generate_user_video(
            duration=float(record_duration),
            fps=10,
            width=800,
            height=380,
            mode=mode,
            speed=speed,
            object_x=object_x,
            force_queue=force_queue,
            destabilize=(destabilize and mode == "naive"),
            logs=logs,
            progress_cb=progress_cb,
            frame_cb=frame_cb,
        )

        if video_bytes:
            st.session_state.user_video = (video_bytes, video_mime)
            progress_box.empty()
        else:
            progress_box.empty()
            st.warning("Could not encode video.")

with button_col3:
    if st.button("Generate scripted 60 s demo"):
        progress = progress_box.progress(0)

        def progress_cb(p):
            progress.progress(int(100 * clamp(p, 0.0, 1.0)))

        def frame_cb(frame):
            live_box.image(bgr_to_rgb(frame))

        video_bytes, video_mime = generate_scripted_video(
            duration=60.0,
            fps=10,
            width=860,
            height=380,
            logs=logs,
            progress_cb=progress_cb,
            frame_cb=frame_cb,
        )

        if video_bytes:
            st.session_state.scripted_video = (video_bytes, video_mime)
            progress_box.empty()
        else:
            progress_box.empty()
            st.warning("Could not encode video.")

if "user_video" in st.session_state:
    video_bytes, video_mime = st.session_state.user_video
    if video_bytes:
        st.subheader("Interactive recorded demo")
        st.video(video_bytes, format=video_mime)

        ext = "mp4" if "mp4" in video_mime else "avi"
        st.download_button(
            "Download interactive demo",
            data=video_bytes,
            file_name=f"wholebody_interactive_demo.{ext}",
            mime=video_mime,
        )

if "scripted_video" in st.session_state:
    video_bytes, video_mime = st.session_state.scripted_video
    if video_bytes:
        st.subheader("Scripted 60 s benchmark demo")
        st.video(video_bytes, format=video_mime)

        ext = "mp4" if "mp4" in video_mime else "avi"
        st.download_button(
            "Download scripted 60 s demo",
            data=video_bytes,
            file_name=f"wholebody_scripted_demo.{ext}",
            mime=video_mime,
        )

st.divider()
st.caption(
    "Benchmark task: walk 2 m while reaching to an object placed along the path. "
    "The coordinated supervisor gates large reaches through double support and raises the CoM-stability weight when margin is low."
)