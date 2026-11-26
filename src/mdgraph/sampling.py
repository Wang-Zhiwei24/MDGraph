"""Numerical sampling diagnostics for production MD trajectories."""

from __future__ import annotations

import numpy as np


def equilibration_check(
    rows: list[dict],
    temperature: float,
    ensemble: str,
    *,
    temperature_tolerance: float,
    volume_tolerance: float,
) -> dict:
    tail = rows[len(rows) // 2 :]
    if len(tail) < 9 or not all(
        np.isfinite(row.get(key, np.nan)) for row in tail for key in ("Temp", "Volume")
    ):
        return {"passed": False, "reason": "Insufficient finite equilibration thermo data"}
    blocks = np.array_split(np.array([[row["Temp"], row["Volume"]] for row in tail]), 3)
    means = np.array([block.mean(axis=0) for block in blocks])
    if np.any(means[:, 1] <= 0):
        return {"passed": False, "reason": "Invalid equilibration volume"}
    temp_deviation = float(np.max(np.abs(means[:, 0] - temperature)) / temperature)
    volume_deviation = float(np.ptp(means[:, 1]) / means[:, 1].mean())
    return {
        "passed": bool(
            temp_deviation <= temperature_tolerance
            and (ensemble != "npt" or volume_deviation <= volume_tolerance)
        ),
        "ensemble": ensemble,
        "block_temperature_k": means[:, 0].tolist(),
        "block_volume_angstrom3": means[:, 1].tolist(),
        "temperature_deviation": temp_deviation,
        "volume_deviation": volume_deviation,
        "temperature_tolerance": temperature_tolerance,
        "volume_tolerance": volume_tolerance if ensemble == "npt" else None,
    }


def block_relative_error(curves: np.ndarray) -> float:
    mean = curves.mean(axis=0)
    scale = np.linalg.norm(mean)
    if scale == 0:
        return 0.0 if not np.any(curves) else float(np.finfo(float).max)
    return float(np.max(np.linalg.norm(curves - mean, axis=1)) / scale)


def windowed_msd(coordinates: np.ndarray) -> np.ndarray:
    """Average squared displacement over every time origin at each lag."""
    count, entities, _ = coordinates.shape
    positions = coordinates - coordinates[0]
    transform = np.fft.rfft(positions, n=2 * count, axis=0)
    power = np.sum(transform.real**2 + transform.imag**2, axis=(1, 2))
    correlation = np.fft.irfft(power, n=2 * count)[:count]
    squared = np.concatenate(([0.0], np.cumsum(np.sum(positions**2, axis=(1, 2)))))
    lags = np.arange(count)
    distances = squared[count - lags] + squared[count] - squared[lags] - 2 * correlation
    return np.maximum(0, distances / ((count - lags) * entities))


def analyze_windowed_msd(
    coordinates: np.ndarray, times_ps: np.ndarray, *, molecular_water: bool, tolerance: float
) -> tuple[list[dict], dict]:
    if len(coordinates) < 60 or not np.allclose(np.diff(times_ps), times_ps[1] - times_ps[0]):
        return [], {
            "passed": False,
            "reason": "MSD needs at least 60 evenly spaced production frames",
        }
    if molecular_water:
        masses = np.array([15.9994, 1.008, 1.008])
        coordinates = (
            coordinates.reshape(len(coordinates), -1, 3, 3) * masses[None, None, :, None]
        ).sum(axis=2)
        coordinates /= masses.sum()
    coordinates = coordinates - coordinates.mean(axis=1, keepdims=True)
    count = len(coordinates) // 3
    lag_count = max(6, count // 3)
    blocks = np.array(
        [windowed_msd(coordinates[i * count : (i + 1) * count])[:lag_count] for i in range(3)]
    )
    average = windowed_msd(coordinates)[:lag_count]
    times = np.arange(lag_count) * (times_ps[1] - times_ps[0])
    error = block_relative_error(blocks[:, 1:])
    sem = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    output = [
        {
            "lag_ps": float(time),
            "msd_angstrom2": float(value),
            "block_sem_angstrom2": float(uncertainty),
        }
        for time, value, uncertainty in zip(times, average, sem)
    ]
    check = {
        "passed": error <= tolerance,
        "block_relative_error": error,
        "tolerance": tolerance,
        "blocks": 3,
        "frames_per_block": count,
        "max_lag_ps": float(times[-1]),
        "tracked_entities": "water molecular centers of mass" if molecular_water else "atoms",
        "method": "multiple time origins with center-of-mass drift removed",
    }
    if molecular_water:
        start = max(1, lag_count // 4)
        interval = slice(start, lag_count)
        exponents = [
            float(np.polyfit(np.log(times[interval]), np.log(block[interval]), 1)[0])
            if np.all(block[interval] > 0)
            else None
            for block in blocks
        ]
        diffusive = all(exponent is not None and 0.7 <= exponent <= 1.3 for exponent in exponents)
        check.update(
            passed=check["passed"] and diffusive,
            diagnostic_window_ps=[float(times[start]), float(times[-1])],
            block_log_slopes=exponents,
            log_slope_range=[0.7, 1.3],
        )
    return output, check
