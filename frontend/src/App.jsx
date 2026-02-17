import React, { useState, useRef, useCallback, useEffect } from 'react';
import ExerciseSelector from './components/ExerciseSelector';
import Webcam from './components/Webcam';
import PoseOverlay from './components/PoseOverlay';
import MetricsPanel from './components/MetricsPanel';
import ImageAnalyzer from './components/ImageAnalyzer';

const BACKEND_PORT = 8001;
const WS_URL = `ws://localhost:${BACKEND_PORT}/ws/pose`;
const API_URL = `http://localhost:${BACKEND_PORT}`;

const EXERCISES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2'];

export default function App() {
    // ── Mode: 'live' or 'image' ──
    const [mode, setMode] = useState('live');

    // ── Live mode state ──
    const [exercise, setExercise] = useState('');
    const [showSkeleton, setShowSkeleton] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [cameraReady, setCameraReady] = useState(false);
    const [cameraError, setCameraError] = useState(null);
    const [metrics, setMetrics] = useState({
        similarity: 0,
        confidence: 0,
        issues: [],
        good: [],
        detected: false,
    });
    const [skeletonColor, setSkeletonColor] = useState('red');
    const [keypoints, setKeypoints] = useState(null);

    // ── Refs ──
    const wsRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const reconnectTimer = useRef(null);
    const exerciseRef = useRef('');
    const mountedRef = useRef(true);
    const readyForFrameRef = useRef(true);

    useEffect(() => {
        exerciseRef.current = exercise;
    }, [exercise]);

    // ── WebSocket connection ──
    const connectWS = useCallback(() => {
        if (wsRef.current) {
            const state = wsRef.current.readyState;
            if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return;
        }

        const ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            setIsConnected(true);
            readyForFrameRef.current = true;
            const ex = exerciseRef.current;
            if (ex) ws.send(JSON.stringify({ exercise: ex }));
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'pose_result') {
                    readyForFrameRef.current = true;
                    setMetrics({
                        similarity: data.similarity,
                        confidence: data.confidence,
                        issues: data.issues || [],
                        good: data.good || [],
                        detected: data.detected,
                    });
                    setSkeletonColor(data.skeleton_color || 'red');
                    setKeypoints(data.detected ? data.keypoints : null);
                }
            } catch { }
        };

        ws.onclose = () => {
            setIsConnected(false);
            if (mountedRef.current) {
                clearTimeout(reconnectTimer.current);
                reconnectTimer.current = setTimeout(() => {
                    if (mountedRef.current) {
                        wsRef.current = null;
                        connectWS();
                    }
                }, 2000);
            }
        };

        ws.onerror = () => { };
        wsRef.current = ws;
    }, []);

    const handleExerciseChange = useCallback(
        async (value) => {
            setExercise(value);
            exerciseRef.current = value;

            if (!value) {
                setIsStreaming(false);
                setMetrics({ similarity: 0, confidence: 0, issues: [], good: [], detected: false });
                setKeypoints(null);
                return;
            }

            try {
                await fetch(`${API_URL}/start_session`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ exercise: value }),
                });
            } catch { }

            setIsStreaming(true);

            const ws = wsRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ exercise: value }));
            } else {
                connectWS();
            }
        },
        [connectWS]
    );

    useEffect(() => {
        mountedRef.current = true;
        if (mode === 'live') connectWS();
        return () => {
            mountedRef.current = false;
            clearTimeout(reconnectTimer.current);
            if (wsRef.current) {
                wsRef.current.onclose = null;
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [connectWS, mode]);

    return (
        <div className="app">
            <header className="app-header">
                <div className="app-logo">
                    <div className="app-logo-icon">🦴</div>
                    <h1>Physio Pose AI</h1>
                </div>

                <div className="app-header-controls">
                    {/* Mode toggle */}
                    <div className="mode-tabs">
                        <button
                            className={`mode-tab ${mode === 'live' ? 'active' : ''}`}
                            onClick={() => setMode('live')}
                        >
                            📹 Live Webcam
                        </button>
                        <button
                            className={`mode-tab ${mode === 'image' ? 'active' : ''}`}
                            onClick={() => setMode('image')}
                        >
                            📸 Upload Image
                        </button>
                    </div>

                    {mode === 'live' && (
                        <>
                            <ExerciseSelector value={exercise} onChange={handleExerciseChange} />

                            <button
                                className={`toggle-btn ${showSkeleton ? 'active' : ''}`}
                                onClick={() => setShowSkeleton((prev) => !prev)}
                            >
                                <span className="toggle-icon">{showSkeleton ? '👁️' : '🚫'}</span>
                                {showSkeleton ? 'Skeleton ON' : 'Skeleton OFF'}
                            </button>

                            <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                                <span className="status-dot" />
                                {isConnected ? 'Connected' : 'Disconnected'}
                            </div>
                        </>
                    )}
                </div>
            </header>

            {mode === 'live' ? (
                <main className="app-main">
                    <section className="webcam-section">
                        <div className="webcam-container">
                            {!exercise ? (
                                <div className="webcam-placeholder">
                                    <div className="webcam-placeholder-icon">🧘</div>
                                    <p>Select an exercise to begin</p>
                                    <span className="hint">Choose from the dropdown above to start real-time pose detection</span>
                                </div>
                            ) : !cameraReady && !cameraError ? (
                                <div className="webcam-placeholder">
                                    <div className="webcam-placeholder-icon">📷</div>
                                    <p>Starting camera...</p>
                                    <span className="hint">Please allow camera access when prompted</span>
                                </div>
                            ) : cameraError ? (
                                <div className="webcam-placeholder">
                                    <div className="webcam-placeholder-icon">❌</div>
                                    <p>Camera Error</p>
                                    <span className="hint">{cameraError}</span>
                                </div>
                            ) : null}

                            <video
                                ref={videoRef}
                                className="webcam-video"
                                autoPlay playsInline muted
                                style={{ display: exercise && cameraReady ? 'block' : 'none' }}
                            />

                            <canvas
                                ref={canvasRef}
                                className="webcam-canvas"
                                style={{ display: exercise && cameraReady ? 'block' : 'none' }}
                            />

                            <Webcam
                                videoRef={videoRef}
                                canvasRef={canvasRef}
                                isStreaming={isStreaming}
                                wsRef={wsRef}
                                readyForFrameRef={readyForFrameRef}
                                onCameraReady={() => { setCameraReady(true); setCameraError(null); }}
                                onCameraError={(msg) => { setCameraError(msg); setCameraReady(false); }}
                            />

                            <PoseOverlay
                                canvasRef={canvasRef}
                                keypoints={keypoints}
                                skeletonColor={skeletonColor}
                                visible={showSkeleton && cameraReady}
                            />
                        </div>

                        {exercise && cameraReady && (
                            <div className="webcam-status-bar">
                                <span>
                                    Exercise: <strong style={{ color: '#8b5cf6' }}>{exercise}</strong>
                                </span>
                                <span className="webcam-fps">
                                    {isConnected ? 'Live • Real-time' : 'Waiting for connection...'}
                                </span>
                            </div>
                        )}
                    </section>

                    {exercise ? (
                        <MetricsPanel metrics={metrics} skeletonColor={skeletonColor} />
                    ) : (
                        <div className="metrics-panel">
                            <div className="start-prompt">
                                <div className="start-prompt-icon">🎯</div>
                                <h2>Ready to Start</h2>
                                <p>Select a yoga exercise from the dropdown to begin real-time pose detection and get instant feedback on your form.</p>
                            </div>
                        </div>
                    )}
                </main>
            ) : (
                <ImageAnalyzer exercises={EXERCISES} />
            )}
        </div>
    );
}
