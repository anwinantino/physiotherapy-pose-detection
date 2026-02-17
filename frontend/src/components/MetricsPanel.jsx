import React from 'react';

export default function MetricsPanel({ metrics, skeletonColor }) {
    const {
        similarity = 0,
        confidence = 0,
        issues = [],
        good = [],
        detected = false,
    } = metrics;

    const simClass = similarity >= 80 ? 'good' : similarity > 0 ? 'bad' : 'neutral';

    return (
        <div className="metrics-panel">
            {/* ── Skeleton Color Indicator ── */}
            {detected && (
                <div className={`skeleton-indicator ${skeletonColor}`}>
                    <span className="skeleton-dot" />
                    {skeletonColor === 'green' ? 'Correct Posture' : 'Needs Correction'}
                </div>
            )}

            {/* ── Similarity Score ── */}
            <div className="metric-card similarity-card">
                <div className="metric-card-header">
                    <div className="metric-card-icon">📊</div>
                    <span className="metric-card-title">Similarity Score</span>
                </div>
                <div className="similarity-value">
                    <span className={`similarity-number ${simClass}`}>
                        {Math.round(similarity)}
                    </span>
                    <span className="similarity-percent">%</span>
                </div>
                <div className="similarity-bar">
                    <div
                        className={`similarity-bar-fill ${simClass}`}
                        style={{ width: `${Math.min(100, similarity)}%` }}
                    />
                </div>
            </div>

            {/* ── Confidence ── */}
            <div className="metric-card confidence-card">
                <div className="metric-card-header">
                    <div className="metric-card-icon">🎯</div>
                    <span className="metric-card-title">Detection Confidence</span>
                </div>
                <div className="confidence-value">{confidence.toFixed(2)}</div>
            </div>

            {/* ── Issues ── */}
            <div className="metric-card issues-card">
                <div className="metric-card-header">
                    <div className="metric-card-icon">⚠️</div>
                    <span className="metric-card-title">
                        Issues {issues.length > 0 && `(${issues.length})`}
                    </span>
                </div>
                {issues.length > 0 ? (
                    <ul className="feedback-list">
                        {issues.map((msg, i) => (
                            <li key={i} className="feedback-item issue">
                                <span className="feedback-icon">✕</span>
                                {msg}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="feedback-empty">No issues detected</p>
                )}
            </div>

            {/* ── Good Posture ── */}
            <div className="metric-card good-card">
                <div className="metric-card-header">
                    <div className="metric-card-icon">✅</div>
                    <span className="metric-card-title">
                        Good {good.length > 0 && `(${good.length})`}
                    </span>
                </div>
                {good.length > 0 ? (
                    <ul className="feedback-list">
                        {good.map((msg, i) => (
                            <li key={i} className="feedback-item good">
                                <span className="feedback-icon">✓</span>
                                {msg}
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="feedback-empty">Perform an exercise to get feedback</p>
                )}
            </div>
        </div>
    );
}
