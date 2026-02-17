"""
FastAPI application — Physio Pose AI backend.

REST endpoints:
    GET  /exercises        → list available exercises
    POST /start_session    → load reference for an exercise
    POST /analyze_image    → upload an image and get full evaluation metrics

WebSocket endpoint (via router):
    WS   /ws/pose          → real-time pose detection stream
"""

import os
import json
import base64

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .websocket_handler import router as ws_router, set_references, _get_engine
from .similarity import compute_similarity, compute_keypoint_similarity, compute_angle_similarity
from .feedback import generate_feedback
from .utils import normalize_keypoints, MIN_CONFIDENCE

# ──────────────────────────────────────────────
# App creation
# ──────────────────────────────────────────────
app = FastAPI(
    title="Physio Pose AI",
    description="Real-time physiotherapy pose detection and feedback",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Load reference data
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(BASE_DIR, "data", "yoga_reference.json")

ALL_EXERCISES = ["downdog", "goddess", "plank", "tree", "warrior2"]

pose_references: dict = {}

if os.path.exists(REFERENCE_PATH):
    with open(REFERENCE_PATH, "r") as f:
        raw = json.load(f)

    # Group samples by pose label
    grouped: dict = {}
    for sample in raw.get("samples", []):
        label = sample["pose_label"]
        if label not in grouped:
            grouped[label] = {"keypoints": [], "angles": []}
        grouped[label]["keypoints"].append(sample["keypoints"])
        grouped[label]["angles"].append(sample["angles"])

    # Compute per-pose median keypoints & angles (median is robust to outliers)
    for label, data in grouped.items():
        avg_kps = np.median(data["keypoints"], axis=0).tolist()

        angle_keys = data["angles"][0].keys()
        avg_angles = {}
        for key in angle_keys:
            vals = [a[key] for a in data["angles"] if a[key] is not None]
            avg_angles[key] = float(np.median(vals)) if vals else None

        pose_references[label] = {
            "keypoints": avg_kps,
            "angles": avg_angles,
        }

    print(f"[Startup] Loaded references for: {list(pose_references.keys())}")
else:
    print(f"[Startup] No reference file found at {REFERENCE_PATH}")
    print("[Startup] Run 'python scripts/build_reference.py' to generate it.")

# Share references with WebSocket handler
set_references(pose_references)

# ──────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────


class SessionRequest(BaseModel):
    exercise: str


@app.get("/exercises")
async def get_exercises():
    """Return the list of available exercises."""
    exercises = list(pose_references.keys()) if pose_references else ALL_EXERCISES
    return {"exercises": exercises}


@app.post("/start_session")
async def start_session(request: SessionRequest):
    """Start a pose detection session for the given exercise."""
    exercise = request.exercise.lower()

    if exercise not in pose_references:
        return {
            "status": "error",
            "message": f"Exercise '{exercise}' not found. "
            f"Available: {list(pose_references.keys())}",
        }

    return {
        "status": "ok",
        "exercise": exercise,
        "reference_angles": pose_references[exercise]["angles"],
    }


@app.post("/analyze_image")
async def analyze_image(
    file: UploadFile = File(...),
    exercise: str = Form(...),
):
    """
    Upload an image and get full pose evaluation metrics.

    Returns detailed analysis including:
    - Pose detection status
    - Similarity score (overall, keypoint, angle breakdowns)
    - Confidence score
    - Joint-by-joint angle analysis
    - Issues and good feedback
    - Detected keypoints
    - Skeleton color
    """
    exercise = exercise.lower()

    if exercise not in pose_references:
        return {
            "status": "error",
            "message": f"Exercise '{exercise}' not found. Available: {list(pose_references.keys())}",
        }

    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"status": "error", "message": "Could not decode image. Please upload a valid JPEG/PNG."}

    # Detect pose
    engine = _get_engine()
    keypoints = engine.detect_keypoints(frame)

    if keypoints is None:
        return {
            "status": "ok",
            "detected": False,
            "message": "No pose detected. Make sure your full body is visible in the image.",
            "similarity": 0,
            "keypoint_similarity": 0,
            "angle_similarity": 0,
            "confidence": 0,
            "skeleton_color": "red",
            "issues": ["No pose detected — make sure your full body is visible"],
            "good": [],
            "keypoints": [],
            "live_angles": {},
            "reference_angles": {},
            "angle_deviations": {},
        }

    # Compute angles and normalize
    live_angles = engine.compute_joint_angles(keypoints)
    live_normalized = normalize_keypoints(keypoints)

    # Confidence
    valid_confs = [kp[2] for kp in keypoints if kp[2] >= MIN_CONFIDENCE]
    mean_confidence = float(np.mean(valid_confs)) if valid_confs else 0.0

    # Reference
    ref = pose_references[exercise]

    # Individual similarity scores
    kp_score = compute_keypoint_similarity(live_normalized, ref["keypoints"])
    angle_score = compute_angle_similarity(live_angles, ref["angles"])
    overall_similarity = compute_similarity(live_normalized, ref["keypoints"], live_angles, ref["angles"])

    # Feedback
    feedback = generate_feedback(live_angles, ref["angles"], overall_similarity, mean_confidence)

    # Angle-by-angle deviation breakdown
    angle_deviations = {}
    for name in live_angles:
        ref_val = ref["angles"].get(name)
        if ref_val is not None and live_angles[name] is not None:
            angle_deviations[name] = {
                "live": round(live_angles[name], 1),
                "reference": round(ref_val, 1),
                "deviation": round(abs(live_angles[name] - ref_val), 1),
                "status": "correct" if abs(live_angles[name] - ref_val) <= 25.0 else "incorrect",
            }

    skeleton_color = "green" if overall_similarity >= 70 else "red"

    # Relative keypoints for frontend display
    h, w = frame.shape[:2]
    relative_kps = [
        [round(kp[0] / w, 5), round(kp[1] / h, 5), round(kp[2], 4)]
        for kp in keypoints
    ]

    return {
        "status": "ok",
        "detected": True,
        "exercise": exercise,
        "similarity": overall_similarity,
        "keypoint_similarity": round(kp_score, 1),
        "angle_similarity": round(angle_score, 1),
        "confidence": round(mean_confidence, 2),
        "skeleton_color": skeleton_color,
        "issues": feedback["issues"],
        "good": feedback["good"],
        "keypoints": relative_kps,
        "live_angles": {k: round(v, 1) if v else None for k, v in live_angles.items()},
        "reference_angles": {k: round(v, 1) if v else None for k, v in ref["angles"].items()},
        "angle_deviations": angle_deviations,
    }


# ──────────────────────────────────────────────
# Mount WebSocket router
# ──────────────────────────────────────────────
app.include_router(ws_router)
