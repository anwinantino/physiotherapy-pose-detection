import React, { useState, useRef } from 'react';

const BACKEND_PORT = 8001;
const API_URL = `http://localhost:${BACKEND_PORT}`;

/**
 * ImageAnalyzer — Upload an image and get full pose evaluation metrics.
 */
export default function ImageAnalyzer({ exercises }) {
    const [exercise, setExercise] = useState('');
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const fileRef = useRef(null);

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setPreview(URL.createObjectURL(file));
        setResult(null);
        setError(null);
    };

    const handleAnalyze = async () => {
        const file = fileRef.current?.files?.[0];
        if (!file || !exercise) return;

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('exercise', exercise);

        try {
            const res = await fetch(`${API_URL}/analyze_image`, {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();
            if (data.status === 'error') {
                setError(data.message);
            } else {
                setResult(data);
            }
        } catch (err) {
            setError('Failed to connect to backend: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const getScoreColor = (val) => {
        if (val >= 80) return '#10b981';
        if (val >= 70) return '#8b5cf6';
        if (val >= 50) return '#f59e0b';
        return '#ef4444';
    };

    return (
        <div className="image-analyzer">
            <div className="analyzer-header">
                <h2>📸 Image Analysis</h2>
                <p className="analyzer-subtitle">Upload a pose image to get detailed evaluation metrics</p>
            </div>

            <div className="analyzer-controls">
                <div className="analyzer-row">
                    <select
                        className="analyzer-select"
                        value={exercise}
                        onChange={(e) => setExercise(e.target.value)}
                    >
                        <option value="">Select exercise...</option>
                        {exercises.map((ex) => (
                            <option key={ex} value={ex}>
                                {ex.charAt(0).toUpperCase() + ex.slice(1).replace(/(\d)/, ' $1')}
                            </option>
                        ))}
                    </select>

                    <label className="analyzer-file-btn">
                        📁 Choose Image
                        <input
                            ref={fileRef}
                            type="file"
                            accept="image/*"
                            onChange={handleFileChange}
                            style={{ display: 'none' }}
                        />
                    </label>

                    <button
                        className="analyzer-run-btn"
                        onClick={handleAnalyze}
                        disabled={!exercise || !preview || loading}
                    >
                        {loading ? '⏳ Analyzing...' : '🔍 Analyze Pose'}
                    </button>
                </div>
            </div>

            <div className="analyzer-body">
                {/* Image Preview */}
                {preview && (
                    <div className="analyzer-preview">
                        <img src={preview} alt="Uploaded pose" />
                    </div>
                )}

                {error && (
                    <div className="analyzer-error">
                        <span>❌</span> {error}
                    </div>
                )}

                {/* Results */}
                {result && (
                    <div className="analyzer-results">
                        {!result.detected ? (
                            <div className="analyzer-error">
                                <span>🚫</span> {result.message || 'No pose detected in image'}
                            </div>
                        ) : (
                            <>
                                {/* Score Cards */}
                                <div className="score-cards">
                                    <div className="score-card main-score" style={{
                                        borderColor: getScoreColor(result.similarity)
                                    }}>
                                        <div className="score-value" style={{
                                            color: getScoreColor(result.similarity)
                                        }}>
                                            {result.similarity}%
                                        </div>
                                        <div className="score-label">Overall Similarity</div>
                                        <div className="score-badge" style={{
                                            background: result.skeleton_color === 'green'
                                                ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                                            color: result.skeleton_color === 'green' ? '#10b981' : '#ef4444',
                                        }}>
                                            {result.skeleton_color === 'green' ? '✅ Good Form' : '⚠️ Needs Work'}
                                        </div>
                                    </div>

                                    <div className="score-card">
                                        <div className="score-value" style={{
                                            color: getScoreColor(result.angle_similarity)
                                        }}>
                                            {result.angle_similarity}%
                                        </div>
                                        <div className="score-label">Angle Accuracy</div>
                                    </div>

                                    <div className="score-card">
                                        <div className="score-value" style={{
                                            color: getScoreColor(result.keypoint_similarity)
                                        }}>
                                            {result.keypoint_similarity}%
                                        </div>
                                        <div className="score-label">Position Accuracy</div>
                                    </div>

                                    <div className="score-card">
                                        <div className="score-value" style={{ color: '#8b5cf6' }}>
                                            {result.confidence}
                                        </div>
                                        <div className="score-label">Detection Confidence</div>
                                    </div>
                                </div>

                                {/* Angle Deviations Table */}
                                {result.angle_deviations && Object.keys(result.angle_deviations).length > 0 && (
                                    <div className="angle-table-wrapper">
                                        <h3>🦴 Joint Angle Analysis</h3>
                                        <table className="angle-table">
                                            <thead>
                                                <tr>
                                                    <th>Joint</th>
                                                    <th>Your Angle</th>
                                                    <th>Reference</th>
                                                    <th>Deviation</th>
                                                    <th>Status</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Object.entries(result.angle_deviations).map(([joint, data]) => (
                                                    <tr key={joint} className={data.status === 'correct' ? 'row-correct' : 'row-incorrect'}>
                                                        <td className="joint-name">
                                                            {joint.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                                        </td>
                                                        <td>{data.live}°</td>
                                                        <td>{data.reference}°</td>
                                                        <td style={{
                                                            color: data.deviation <= 15 ? '#10b981'
                                                                : data.deviation <= 30 ? '#f59e0b' : '#ef4444'
                                                        }}>
                                                            {data.deviation}°
                                                        </td>
                                                        <td>
                                                            <span className={`status-pill ${data.status}`}>
                                                                {data.status === 'correct' ? '✅' : '❌'}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}

                                {/* Feedback Lists */}
                                <div className="feedback-columns">
                                    {result.good.length > 0 && (
                                        <div className="feedback-col good-col">
                                            <h3>✅ Correct</h3>
                                            <ul>
                                                {result.good.map((msg, i) => (
                                                    <li key={i}>{msg}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {result.issues.length > 0 && (
                                        <div className="feedback-col issues-col">
                                            <h3>⚠️ Issues</h3>
                                            <ul>
                                                {result.issues.map((msg, i) => (
                                                    <li key={i}>{msg}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
