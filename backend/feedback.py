"""
Feedback generation engine.
Compares live joint angles against reference and produces
human-readable issues and positive feedback.
"""

from .utils import ANGLE_DEFINITIONS

# ──────────────────────────────────────────────
# Human-readable feedback templates
# ──────────────────────────────────────────────
ANGLE_ISSUE_MESSAGES = {
    "left_elbow": {
        "too_small": "Left elbow bent too much",
        "too_large": "Left elbow not bent enough",
    },
    "right_elbow": {
        "too_small": "Right elbow bent too much",
        "too_large": "Right elbow not bent enough",
    },
    "left_shoulder": {
        "too_small": "Left arm too close to body",
        "too_large": "Left arm raised too high",
    },
    "right_shoulder": {
        "too_small": "Right arm too close to body",
        "too_large": "Right arm raised too high",
    },
    "left_knee": {
        "too_small": "Left knee bent too much",
        "too_large": "Left knee not bent enough",
    },
    "right_knee": {
        "too_small": "Right knee bent too much",
        "too_large": "Right knee not bent enough",
    },
    "left_hip": {
        "too_small": "Left hip angle too narrow",
        "too_large": "Left hip angle too wide",
    },
    "right_hip": {
        "too_small": "Right hip angle too narrow",
        "too_large": "Right hip angle too wide",
    },
}

GOOD_MESSAGES = {
    "left_elbow": "Left elbow position correct",
    "right_elbow": "Right elbow position correct",
    "left_shoulder": "Left shoulder alignment correct",
    "right_shoulder": "Right shoulder alignment correct",
    "left_knee": "Left knee position correct",
    "right_knee": "Right knee position correct",
    "left_hip": "Left hip alignment correct",
    "right_hip": "Right hip alignment correct",
}

# Tolerance in degrees for "correct" classification
# 25° is realistic for webcam + real-time detection noise
ANGLE_TOLERANCE = 25.0


def generate_feedback(
    live_angles: dict,
    ref_angles: dict,
    similarity: float,
    confidence: float,
) -> dict:
    """
    Generate detailed feedback comparing live pose angles to reference.

    Args:
        live_angles:  Dict of live joint angles.
        ref_angles:   Dict of reference joint angles.
        similarity:   Overall similarity score (0–100).
        confidence:   Mean keypoint confidence (0–1).

    Returns:
        {
            "similarity": float,
            "confidence": float,
            "issues": [str, ...],
            "good": [str, ...],
        }
    """
    issues = []
    good = []

    for name in ANGLE_DEFINITIONS:
        live_val = live_angles.get(name)
        ref_val = ref_angles.get(name)

        if live_val is None or ref_val is None:
            continue

        diff = live_val - ref_val

        if abs(diff) <= ANGLE_TOLERANCE:
            good.append(GOOD_MESSAGES[name])
        elif diff < -ANGLE_TOLERANCE:
            issues.append(ANGLE_ISSUE_MESSAGES[name]["too_small"])
        else:
            issues.append(ANGLE_ISSUE_MESSAGES[name]["too_large"])

    return {
        "similarity": similarity,
        "confidence": round(confidence, 2),
        "issues": issues,
        "good": good,
    }
