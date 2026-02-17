"""
Pose detection engine using MediaPipe PoseLandmarker (Task API).
Compatible with mediapipe >= 0.10.x.

Provides COCO-17 keypoint detection, angle computation,
skeleton drawing, and EMA pose smoothing.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .utils import (
    MEDIAPIPE_INDICES,
    COCO_SKELETON,
    ANGLE_DEFINITIONS,
    MIN_CONFIDENCE,
    compute_angle,
)

# Path to model file (relative to project root)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_HEAVY = os.path.join(_BASE_DIR, "models", "pose_landmarker_heavy.task")
_MODEL_LITE = os.path.join(_BASE_DIR, "models", "pose_landmarker_lite.task")


def _get_model_path(prefer_heavy: bool = False) -> str:
    """Return available model path, preferring lite for speed."""
    if prefer_heavy and os.path.exists(_MODEL_HEAVY):
        return _MODEL_HEAVY
    if os.path.exists(_MODEL_LITE):
        return _MODEL_LITE
    if os.path.exists(_MODEL_HEAVY):
        return _MODEL_HEAVY
    raise FileNotFoundError(
        f"No PoseLandmarker model found. Expected at:\n"
        f"  {_MODEL_LITE}\n  {_MODEL_HEAVY}\n"
        f"Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models"
    )


class PoseEngine:
    """Real-time pose detection engine using MediaPipe PoseLandmarker."""

    def __init__(self, use_heavy: bool = False):
        model_path = _get_model_path(prefer_heavy=use_heavy)
        print(f"[PoseEngine] Using model: {os.path.basename(model_path)}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._prev_keypoints = None
        self._smooth_alpha = 0.6

    def detect_keypoints(self, frame: np.ndarray) -> list | None:
        """
        Detect COCO-17 keypoints from a BGR frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of 17 [x, y, confidence] in pixel coordinates,
            or None if no pose detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks[0]  # first person

        keypoints = []
        for coco_idx in range(17):
            mp_idx = MEDIAPIPE_INDICES[coco_idx]
            lm = landmarks[mp_idx]
            keypoints.append([lm.x * w, lm.y * h, float(lm.visibility)])

        return keypoints

    def compute_joint_angles(self, keypoints: list) -> dict:
        """
        Compute 8 joint angles from COCO-17 keypoints.

        Skips any angle where a constituent joint has
        confidence < MIN_CONFIDENCE.

        Returns:
            Dict mapping angle name → degrees (or None).
        """
        angles = {}
        for name, (i, j, k) in ANGLE_DEFINITIONS.items():
            p1, p2, p3 = keypoints[i], keypoints[j], keypoints[k]
            if (
                p1[2] < MIN_CONFIDENCE
                or p2[2] < MIN_CONFIDENCE
                or p3[2] < MIN_CONFIDENCE
            ):
                angles[name] = None
            else:
                angles[name] = round(compute_angle(p1, p2, p3), 2)
        return angles

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: list,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        radius: int = 5,
    ) -> np.ndarray:
        """Draw COCO-17 skeleton on a BGR frame."""
        for i, j in COCO_SKELETON:
            if keypoints[i][2] >= MIN_CONFIDENCE and keypoints[j][2] >= MIN_CONFIDENCE:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        for kp in keypoints:
            if kp[2] >= MIN_CONFIDENCE:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), radius, color, -1, cv2.LINE_AA)

        return frame

    def smooth_pose(self, current: list) -> list:
        """Apply Exponential Moving Average (EMA) smoothing."""
        if self._prev_keypoints is None:
            self._prev_keypoints = [kp[:] for kp in current]
            return current

        alpha = self._smooth_alpha
        smoothed = []
        for i in range(len(current)):
            if current[i][2] < MIN_CONFIDENCE:
                smoothed.append(current[i][:])
            else:
                sx = alpha * current[i][0] + (1 - alpha) * self._prev_keypoints[i][0]
                sy = alpha * current[i][1] + (1 - alpha) * self._prev_keypoints[i][1]
                smoothed.append([sx, sy, current[i][2]])

        self._prev_keypoints = [kp[:] for kp in smoothed]
        return smoothed

    def close(self):
        """Release resources."""
        self._landmarker.close()
