# Physio Pose AI — Real-time Physiotherapy Pose Detection

AI-powered physiotherapy assistant using **MediaPipe BlazePose** for real-time pose detection, with a **FastAPI** WebSocket backend and **React** frontend.

## Features

- 🎯 **5 Yoga Exercises**: Downward Dog, Goddess, Plank, Tree, Warrior II
- 🦴 **Real-time Skeleton Overlay**: COCO-17 keypoints drawn on webcam feed
- 🟢🔴 **Color-coded Posture**: Green (correct) / Red (needs correction)
- 📊 **Similarity Score**: 0–100% weighted comparison (60% keypoints + 40% angles)
- 💬 **Live Feedback**: Issues list + correct posture confirmations
- 👁️ **Toggle Skeleton**: Show/hide overlay with one click
- ⚡ **<200ms Latency**: MediaPipe BlazePose on CPU

---

## Quick Setup

### 1. Install Python Dependencies

```bash
# From project root
pip install -r requirements.txt
```

### 2. Generate Reference Poses

```bash
python scripts/build_reference.py
```

This processes all images in `DATASET/TRAIN/` and saves `data/yoga_reference.json`.

### 3. Start Backend

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Install & Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Open Browser

Go to **http://localhost:5173**

1. Select an exercise from the dropdown
2. Allow camera access
3. Strike the pose — see real-time skeleton + feedback!

---

## Project Structure

```
physiotherapy-pose-detection/
├── DATASET/TRAIN/{downdog,goddess,plank,tree,warrior2}/
├── data/yoga_reference.json          ← generated
├── backend/
│   ├── app.py                        ← FastAPI app + REST endpoints
│   ├── pose_engine.py                ← MediaPipe BlazePose wrapper
│   ├── similarity.py                 ← Weighted similarity scoring
│   ├── feedback.py                   ← Human-readable feedback
│   ├── websocket_handler.py          ← WebSocket real-time handler
│   └── utils.py                      ← COCO-17 constants, normalization
├── frontend/src/
│   ├── App.jsx                       ← Main app + WebSocket lifecycle
│   ├── components/
│   │   ├── ExerciseSelector.jsx      ← Dropdown selector
│   │   ├── Webcam.jsx                ← Camera capture + frame streaming
│   │   ├── PoseOverlay.jsx           ← Canvas skeleton overlay
│   │   └── MetricsPanel.jsx          ← Scores + feedback display
│   └── index.css                     ← Dark glassmorphism theme
├── scripts/build_reference.py        ← Reference JSON builder
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer     | Tech                        |
|-----------|-----------------------------|
| Pose AI   | MediaPipe BlazePose (COCO-17) |
| Backend   | FastAPI + WebSocket         |
| Frontend  | React + Vite                |
| Streaming | WebSocket (base64 JPEG)     |
| Styling   | Vanilla CSS (glassmorphism) |

## API Reference

| Endpoint           | Method    | Description                      |
|--------------------|-----------|----------------------------------|
| `/exercises`       | GET       | List available exercises         |
| `/start_session`   | POST      | Load reference for an exercise   |
| `/ws/pose`         | WebSocket | Real-time pose detection stream  |
