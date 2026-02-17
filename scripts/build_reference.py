"""
Build reference JSON from training images using MediaPipe PoseLandmarker.

Processes all images in DATASET/TRAIN/{pose_label}/ and generates
data/yoga_reference.json with normalized keypoints, joint angles,
and mean confidence for each sample.

Usage:
    python scripts/build_reference.py
"""

import os
import sys
import json
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.utils import (
    MEDIAPIPE_INDICES,
    ANGLE_DEFINITIONS,
    MIN_CONFIDENCE,
    compute_angle,
    normalize_keypoints,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_reference():
    """Process training images and build the reference JSON."""
    dataset_dir = os.path.join(PROJECT_ROOT, "DATASET", "TRAIN")
    output_path = os.path.join(PROJECT_ROOT, "data", "yoga_reference.json")

    # Find model
    model_heavy = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_heavy.task")
    model_lite = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_lite.task")

    model_path = None
    if os.path.exists(model_heavy):
        model_path = model_heavy
    elif os.path.exists(model_lite):
        model_path = model_lite
    else:
        print("ERROR: No PoseLandmarker model found in models/ directory.")
        print("Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models")
        print(f"Expected at: {model_heavy}")
        sys.exit(1)

    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize PoseLandmarker in IMAGE mode for best accuracy
    print(f"Using model: {os.path.basename(model_path)}")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    samples = []
    pose_labels = sorted(
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )

    print(f"Found pose labels: {pose_labels}")
    print(f"Output: {output_path}\n")

    total_processed = 0
    total_skipped = 0
    start_time = time.time()

    for label in pose_labels:
        label_dir = os.path.join(dataset_dir, label)
        image_files = sorted(
            f
            for f in os.listdir(label_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

        label_count = 0
        label_skipped = 0

        print(f"[{label}] Processing {len(image_files)} images...", end=" ", flush=True)

        for img_name in image_files:
            img_path = os.path.join(label_dir, img_name)

            image = cv2.imread(img_path)
            if image is None:
                label_skipped += 1
                continue

            # Convert to MediaPipe Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect pose
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                label_skipped += 1
                continue

            landmarks = result.pose_landmarks[0]

            # Extract COCO-17 keypoints (normalized 0–1 coords)
            keypoints = []
            for coco_idx in range(17):
                mp_idx = MEDIAPIPE_INDICES[coco_idx]
                lm = landmarks[mp_idx]
                keypoints.append([float(lm.x), float(lm.y), float(lm.visibility)])

            # Normalize relative to torso center
            normalized = normalize_keypoints(keypoints)

            # Compute 8 joint angles
            angles = {}
            for name, (i, j, k) in ANGLE_DEFINITIONS.items():
                p1, p2, p3 = keypoints[i], keypoints[j], keypoints[k]
                if (
                    p1[2] >= MIN_CONFIDENCE
                    and p2[2] >= MIN_CONFIDENCE
                    and p3[2] >= MIN_CONFIDENCE
                ):
                    angles[name] = round(compute_angle(p1, p2, p3), 2)
                else:
                    angles[name] = None

            # Mean confidence of valid keypoints
            valid_confs = [kp[2] for kp in keypoints if kp[2] >= MIN_CONFIDENCE]
            mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

            sample = {
                "image_name": img_name,
                "pose_label": label,
                "keypoints": [[round(v, 6) for v in kp] for kp in normalized],
                "angles": angles,
                "confidence_mean": round(mean_conf, 4),
            }
            samples.append(sample)
            label_count += 1

        total_processed += label_count
        total_skipped += label_skipped
        print(f"OK {label_count}, skipped {label_skipped}")

    landmarker.close()

    # Write output
    output = {
        "dataset": "yoga_pose_reference",
        "num_keypoints": 17,
        "samples": samples,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Total samples: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Poses: {sorted(set(s['pose_label'] for s in samples))}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    build_reference()
