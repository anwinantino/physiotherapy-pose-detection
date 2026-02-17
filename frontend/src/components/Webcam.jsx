import React, { useRef, useEffect, useCallback } from 'react';

/**
 * Webcam component — captures video frames and sends them
 * over WebSocket as base64 JPEG.
 *
 * Uses backpressure: only sends a new frame when the backend
 * has responded to the previous one (via readyForFrameRef).
 */
export default function Webcam({
    videoRef,
    canvasRef,
    isStreaming,
    wsRef,
    readyForFrameRef,
    onCameraReady,
    onCameraError,
}) {
    const captureCanvasRef = useRef(null);
    const animFrameRef = useRef(null);

    // Minimum interval between sends (caps at ~15fps even if backend is fast)
    const MIN_INTERVAL_MS = 66;
    const lastSendRef = useRef(0);

    // Start webcam
    useEffect(() => {
        let stream = null;

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user',
                    },
                    audio: false,
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    await videoRef.current.play();
                    onCameraReady?.();
                }
            } catch (err) {
                console.error('Camera error:', err);
                onCameraError?.(err.message || 'Camera access denied');
            }
        }

        if (isStreaming) {
            startCamera();
        }

        return () => {
            if (stream) {
                stream.getTracks().forEach((t) => t.stop());
            }
        };
    }, [isStreaming]);

    // Frame capture loop
    const captureLoop = useCallback(() => {
        const video = videoRef.current;
        const captureCanvas = captureCanvasRef.current;
        const ws = wsRef.current;

        if (!video || !captureCanvas || video.readyState < 2) {
            animFrameRef.current = requestAnimationFrame(captureLoop);
            return;
        }

        const now = performance.now();
        const elapsed = now - lastSendRef.current;

        // Only send if:
        // 1. Enough time has passed (rate limit)
        // 2. WebSocket is open
        // 3. Backend has responded to the previous frame (backpressure)
        if (
            elapsed >= MIN_INTERVAL_MS &&
            ws &&
            ws.readyState === WebSocket.OPEN &&
            readyForFrameRef.current
        ) {
            lastSendRef.current = now;

            // Set canvas size to match video (only on first frame or resolution change)
            if (captureCanvas.width !== video.videoWidth) {
                captureCanvas.width = video.videoWidth;
                captureCanvas.height = video.videoHeight;
            }

            // Sync overlay canvas dimensions
            if (canvasRef?.current) {
                if (canvasRef.current.width !== video.videoWidth) {
                    canvasRef.current.width = video.videoWidth;
                    canvasRef.current.height = video.videoHeight;
                }
            }

            const ctx = captureCanvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            // Mark as waiting for response (backpressure)
            readyForFrameRef.current = false;

            // Send as base64 JPEG (quality=0.6 for speed)
            const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.6);
            try {
                ws.send(JSON.stringify({ frame: dataUrl }));
            } catch (e) {
                // If send fails, allow next attempt
                readyForFrameRef.current = true;
            }
        }

        animFrameRef.current = requestAnimationFrame(captureLoop);
    }, [videoRef, canvasRef, wsRef, readyForFrameRef]);

    // Start/stop capture loop
    useEffect(() => {
        if (isStreaming) {
            animFrameRef.current = requestAnimationFrame(captureLoop);
        }

        return () => {
            if (animFrameRef.current) {
                cancelAnimationFrame(animFrameRef.current);
            }
        };
    }, [isStreaming, captureLoop]);

    return (
        <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
    );
}
