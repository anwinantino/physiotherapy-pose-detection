"""
Utility functions and constants for pose detection.
Handles COCO-17 keypoint format, angle computation, and normalization.
"""

import numpy as np
import math

# ──────────────────────────────────────────────
# COCO-17 Keypoint names (index = keypoint ID)
# ──────────────────────────────────────────────
COCO_KEYPOINTS = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# ──────────────────────────────────────────────
# MediaPipe → COCO-17 mapping
# Index = COCO-17 keypoint index
# Value = MediaPipe landmark index
# ──────────────────────────────────────────────
MEDIAPIPE_INDICES = [
    0,   # COCO 0  nose       → MP 0
    2,   # COCO 1  left_eye   → MP 2
    5,   # COCO 2  right_eye  → MP 5
    7,   # COCO 3  left_ear   → MP 7
    8,   # COCO 4  right_ear  → MP 8
    11,  # COCO 5  left_shoulder  → MP 11
    12,  # COCO 6  right_shoulder → MP 12
    13,  # COCO 7  left_elbow     → MP 13
    14,  # COCO 8  right_elbow    → MP 14
    15,  # COCO 9  left_wrist     → MP 15
    16,  # COCO 10 right_wrist    → MP 16
    23,  # COCO 11 left_hip       → MP 23
    24,  # COCO 12 right_hip      → MP 24
    25,  # COCO 13 left_knee      → MP 25
    26,  # COCO 14 right_knee     → MP 26
    27,  # COCO 15 left_ankle     → MP 27
    28,  # COCO 16 right_ankle    → MP 28
]

# ──────────────────────────────────────────────
# COCO-17 skeleton connections for drawing
# Each tuple = (keypoint_A, keypoint_B)
# ──────────────────────────────────────────────
COCO_SKELETON = [
    (0, 1),   (0, 2),              # nose → eyes
    (1, 3),   (2, 4),              # eyes → ears
    (5, 6),                        # shoulder → shoulder
    (5, 7),   (7, 9),              # left arm
    (6, 8),   (8, 10),             # right arm
    (5, 11),  (6, 12),             # torso sides
    (11, 12),                      # hip → hip
    (11, 13), (13, 15),            # left leg
    (12, 14), (14, 16),            # right leg
]

# ──────────────────────────────────────────────
# Joint angle definitions
# (point_A, vertex, point_B) — angle measured at vertex
# ──────────────────────────────────────────────
ANGLE_DEFINITIONS = {
    "left_elbow":     (5, 7, 9),     # shoulder → elbow → wrist
    "right_elbow":    (6, 8, 10),    # shoulder → elbow → wrist
    "left_shoulder":  (7, 5, 11),    # elbow → shoulder → hip
    "right_shoulder": (8, 6, 12),    # elbow → shoulder → hip
    "left_knee":      (11, 13, 15),  # hip → knee → ankle
    "right_knee":     (12, 14, 16),  # hip → knee → ankle
    "left_hip":       (5, 11, 13),   # shoulder → hip → knee
    "right_hip":      (6, 12, 14),   # shoulder → hip → knee
}

# Minimum confidence to consider a keypoint valid
MIN_CONFIDENCE = 0.3


def compute_angle(p1: list, p2: list, p3: list) -> float:
    """
    Compute the angle at p2 formed by the vectors p1→p2 and p3→p2.

    Args:
        p1, p2, p3: Points as [x, y] or [x, y, conf].

    Returns:
        Angle in degrees (0–180).
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def normalize_keypoints(keypoints: list) -> list:
    """
    Normalize keypoints relative to torso center and torso scale.

    - Translates so midpoint(left_hip, right_hip) = origin
    - Scales by the distance from left_shoulder to right_hip

    Args:
        keypoints: list of 17 [x, y, conf] entries.

    Returns:
        Normalized keypoints (same format).
    """
    kps = np.array(keypoints, dtype=np.float64)  # (17, 3)

    # Torso center = midpoint of hips
    left_hip = kps[11, :2]
    right_hip = kps[12, :2]
    torso_center = (left_hip + right_hip) / 2.0

    # Scale factor = distance from left_shoulder to right_hip
    left_shoulder = kps[5, :2]
    torso_size = np.linalg.norm(left_shoulder - right_hip)
    if torso_size < 1e-8:
        torso_size = 1.0  # prevent division by zero

    normalized = kps.copy()
    normalized[:, 0] = (kps[:, 0] - torso_center[0]) / torso_size
    normalized[:, 1] = (kps[:, 1] - torso_center[1]) / torso_size
    # confidence stays unchanged

    return normalized.tolist()
