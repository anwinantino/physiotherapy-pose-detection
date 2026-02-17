import React from 'react';

/**
 * COCO-17 skeleton connections for canvas drawing.
 * Each pair = [keypointA_index, keypointB_index]
 */
const SKELETON_CONNECTIONS = [
    [0, 1], [0, 2],       // nose → eyes
    [1, 3], [2, 4],       // eyes → ears
    [5, 6],               // shoulders
    [5, 7], [7, 9],       // left arm
    [6, 8], [8, 10],      // right arm
    [5, 11], [6, 12],     // torso
    [11, 12],             // hips
    [11, 13], [13, 15],   // left leg
    [12, 14], [14, 16],   // right leg
];

const MIN_CONFIDENCE = 0.3;

export default function PoseOverlay({ canvasRef, keypoints, skeletonColor, visible }) {
    React.useEffect(() => {
        const canvas = canvasRef?.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        // Clear previous frame
        ctx.clearRect(0, 0, w, h);

        if (!visible || !keypoints || keypoints.length !== 17) return;

        const color = skeletonColor === 'green' ? '#22c55e' : '#ef4444';
        const glowColor = skeletonColor === 'green'
            ? 'rgba(34, 197, 94, 0.4)'
            : 'rgba(239, 68, 68, 0.4)';

        // Convert relative (0–1) keypoints to pixel coords
        const pts = keypoints.map(([x, y, conf]) => ({
            x: x * w,
            y: y * h,
            conf,
        }));

        // ── Draw limb connections ──
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';

        // Glow effect
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 8;

        for (const [i, j] of SKELETON_CONNECTIONS) {
            if (pts[i].conf >= MIN_CONFIDENCE && pts[j].conf >= MIN_CONFIDENCE) {
                ctx.beginPath();
                ctx.moveTo(pts[i].x, pts[i].y);
                ctx.lineTo(pts[j].x, pts[j].y);
                ctx.stroke();
            }
        }

        // ── Draw keypoint dots ──
        ctx.shadowBlur = 12;
        for (const pt of pts) {
            if (pt.conf >= MIN_CONFIDENCE) {
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
                ctx.fill();

                // White center dot
                ctx.fillStyle = '#fff';
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Reset shadow
        ctx.shadowBlur = 0;
    }, [canvasRef, keypoints, skeletonColor, visible]);

    return null; // Rendering happens on the shared canvas via ref
}
