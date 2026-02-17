"""
Similarity computation engine.
Compares live pose angles against reference poses.

Key insight: joint ANGLES are the most reliable signal because
they are invariant to body size, camera distance, and position.
Keypoint positions (even normalized) suffer from averaging artifacts.
"""

import numpy as np
from .utils import ANGLE_DEFINITIONS, MIN_CONFIDENCE


def compute_keypoint_similarity(live_kps: list, ref_kps: list) -> float:
    """
    Compute similarity based on normalized keypoint distances.
    Uses a soft Gaussian-like scoring so small deviations don't destroy the score.

    Returns:
        Score in range [0, 100].
    """
    live = np.array(live_kps)
    ref = np.array(ref_kps)

    # Only compare keypoints where both have sufficient confidence
    mask = (live[:, 2] >= MIN_CONFIDENCE) & (ref[:, 2] >= MIN_CONFIDENCE)

    if mask.sum() < 3:  # need at least 3 keypoints for meaningful comparison
        return 50.0  # neutral / unknown

    live_pts = live[mask, :2]
    ref_pts = ref[mask, :2]

    distances = np.linalg.norm(live_pts - ref_pts, axis=1)

    # Softer exponential decay: k=2.0 (was 4.0)
    # This means a distance of 0.35 (normalized units) still gives ~50%
    k = 2.0
    similarities = np.exp(-k * distances)

    return float(np.mean(similarities) * 100)


def compute_angle_similarity(live_angles: dict, ref_angles: dict) -> float:
    """
    Compute similarity based on angular deviations.
    Uses a softer scoring curve with tolerance band.

    Returns:
        Score in range [0, 100].
    """
    scores = []

    for name in ANGLE_DEFINITIONS:
        live_angle = live_angles.get(name)
        ref_angle = ref_angles.get(name)

        if live_angle is None or ref_angle is None:
            continue

        diff = abs(live_angle - ref_angle)

        # Tolerance band: 0-15° = 100%, then soft falloff
        if diff <= 15.0:
            score = 100.0
        elif diff <= 30.0:
            # Gradual falloff from 100% to 70%
            score = 100.0 - (diff - 15.0) * 2.0  # 30° diff = 70%
        elif diff <= 60.0:
            # Slower falloff from 70% to 30%
            score = 70.0 - (diff - 30.0) * (40.0 / 30.0)  # 60° diff = 30%
        else:
            # Harsh falloff below 30%
            score = max(0.0, 30.0 - (diff - 60.0))

        scores.append(score)

    if not scores:
        return 50.0

    return float(np.mean(scores))


def compute_similarity(
    live_kps: list,
    ref_kps: list,
    live_angles: dict,
    ref_angles: dict,
) -> float:
    """
    Compute final weighted similarity score.

    Angles are weighted much more heavily (75%) because they are
    invariant to body proportions and camera position.
    Keypoint positions provide a secondary spatial check (25%).

    Returns:
        Clamped score in [0, 100], rounded to 1 decimal.
    """
    kp_score = compute_keypoint_similarity(live_kps, ref_kps)
    angle_score = compute_angle_similarity(live_angles, ref_angles)

    # Angles are the reliable signal → 75% weight
    final = 0.25 * kp_score + 0.75 * angle_score
    return round(max(0.0, min(100.0, final)), 1)
