import React from 'react';

const EXERCISES = [
    { value: '', label: 'Select Exercise...' },
    { value: 'downdog', label: 'Downward Dog' },
    { value: 'goddess', label: 'Goddess' },
    { value: 'plank', label: 'Plank' },
    { value: 'tree', label: 'Tree' },
    { value: 'warrior2', label: 'Warrior II' },
];

export default function ExerciseSelector({ value, onChange, disabled }) {
    return (
        <div className="exercise-selector">
            <label htmlFor="exercise-select">Exercise</label>
            <select
                id="exercise-select"
                className="exercise-select"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                disabled={disabled}
            >
                {EXERCISES.map((ex) => (
                    <option key={ex.value} value={ex.value}>
                        {ex.label}
                    </option>
                ))}
            </select>
        </div>
    );
}
