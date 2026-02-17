"""
WebSocket handler for real-time pose detection streaming.
Uses asyncio thread executor to avoid blocking the event loop during detection.
"""

import asyncio
import base64
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .pose_engine import PoseEngine
from .similarity import compute_similarity
from .feedback import generate_feedback
from .utils import normalize_keypoints, MIN_CONFIDENCE

router = APIRouter()

# Module-level state
_pose_references: dict = {}
_engine: PoseEngine | None = None

# Single-threaded executor so MediaPipe calls don't overlap
_executor = ThreadPoolExecutor(max_workers=1)


def set_references(refs: dict):
    """Set the computed pose references for comparison."""
    global _pose_references
    _pose_references = refs


def _get_engine() -> PoseEngine:
    """Lazy-init a shared PoseEngine singleton."""
    global _engine
    if _engine is None:
        _engine = PoseEngine(use_heavy=False)
    return _engine


def _process_frame(engine: PoseEngine, frame: np.ndarray, current_exercise: str | None):
    """
    Synchronous frame processing — runs in thread executor.
    Returns the complete result dict to send back over WebSocket.
    """
    keypoints = engine.detect_keypoints(frame)

    if keypoints is None:
        return {
            "type": "pose_result",
            "detected": False,
            "skeleton_color": "red",
            "similarity": 0,
            "confidence": 0,
            "issues": ["No pose detected — make sure your full body is visible"],
            "good": [],
            "keypoints": [],
        }

    # Smooth pose (EMA)
    keypoints = engine.smooth_pose(keypoints)

    # Compute joint angles
    live_angles = engine.compute_joint_angles(keypoints)

    # Normalize for comparison
    live_normalized = normalize_keypoints(keypoints)

    # Mean confidence
    valid_confs = [kp[2] for kp in keypoints if kp[2] >= MIN_CONFIDENCE]
    mean_confidence = float(np.mean(valid_confs)) if valid_confs else 0.0

    # Compare with reference
    similarity = 0.0
    feedback_result = {
        "similarity": 0,
        "confidence": round(mean_confidence, 2),
        "issues": ["Select an exercise to start pose comparison"],
        "good": [],
    }

    if current_exercise and current_exercise in _pose_references:
        ref = _pose_references[current_exercise]
        similarity = compute_similarity(
            live_normalized,
            ref["keypoints"],
            live_angles,
            ref["angles"],
        )
        feedback_result = generate_feedback(
            live_angles, ref["angles"], similarity, mean_confidence
        )

    skeleton_color = "green" if similarity >= 70 else "red"

    # Convert keypoints to relative (0–1) coordinates for the frontend
    h, w = frame.shape[:2]
    relative_kps = [
        [round(kp[0] / w, 5), round(kp[1] / h, 5), round(kp[2], 4)]
        for kp in keypoints
    ]

    return {
        "type": "pose_result",
        "detected": True,
        "skeleton_color": skeleton_color,
        "similarity": feedback_result["similarity"],
        "confidence": feedback_result["confidence"],
        "issues": feedback_result["issues"],
        "good": feedback_result["good"],
        "keypoints": relative_kps,
    }


@router.websocket("/ws/pose")
async def websocket_pose(websocket: WebSocket):
    """Real-time pose detection WebSocket endpoint."""
    await websocket.accept()
    print("[WS] Client connected")

    try:
        engine = _get_engine()
    except Exception as e:
        print(f"[WS] Failed to init PoseEngine: {e}")
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    current_exercise = None
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_json()

            # ── Exercise selection ──
            if "exercise" in data:
                current_exercise = data["exercise"].lower()
                print(f"[WS] Exercise set: {current_exercise}")
                await websocket.send_json(
                    {"type": "session_started", "exercise": current_exercise}
                )
                continue

            # ── Frame processing ──
            if "frame" not in data:
                continue

            frame_data = data["frame"]
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            # Run detection in thread executor so we don't block the event loop
            # (keeps WebSocket ping/pong alive)
            try:
                result = await loop.run_in_executor(
                    _executor, _process_frame, engine, frame, current_exercise
                )
                await websocket.send_json(result)
            except Exception as e:
                print(f"[WS] Processing error: {e}")
                continue

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        traceback.print_exc()
