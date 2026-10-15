"""Shared operators and conclusion models for historian pipeline audits.

All temporal quantities are expressed in physical minutes and converted to an
integer sampling-grid offset with an explicit compatibility check.  The module
contains no process-specific population, rollout, or interpretation logic.
"""

from __future__ import annotations

import csv
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MODEL_ORDER = ("PERSISTENCE", "RIDGE_AR", "SHALLOW_GBR")


def minutes_to_steps(minutes: float, dt_minutes: float) -> int:
    if dt_minutes <= 0 or minutes < 0:
        raise ValueError("minutes and dt must define a non-negative offset")
    raw = float(minutes) / float(dt_minutes)
    steps = int(round(raw))
    if not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{minutes:g} min is not representable on dt={dt_minutes:g} min")
    return steps


def periodic_indices(n: int, stride: int) -> np.ndarray:
    if n < 2 or stride < 1:
        raise ValueError("periodic retention requires n>=2 and stride>=1")
    indices = list(range(0, int(n), int(stride)))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return np.asarray(indices, dtype=int)


def swinging_door_indices(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Slope-corridor swinging-door retention with endpoint preservation."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("swinging-door retention requires at least two samples")
    if not np.all(np.isfinite(x)) or epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("invalid swinging-door input")
    kept = [0]
    anchor = 0
    lower = -np.inf
    upper = np.inf
    i = 1
    while i < n:
        width = i - anchor
        candidate_lower = (x[i] - epsilon - x[anchor]) / width
        candidate_upper = (x[i] + epsilon - x[anchor]) / width
        new_lower = max(lower, candidate_lower)
        new_upper = min(upper, candidate_upper)
        if new_lower > new_upper:
            retain = i - 1
            if retain <= anchor:
                retain = i
            if retain != kept[-1]:
                kept.append(retain)
            anchor = retain
            lower = -np.inf
            upper = np.inf
            if anchor == i:
                i += 1
            continue
        lower = new_lower
        upper = new_upper
        i += 1
    if kept[-1] != n - 1:
        kept.append(n - 1)
    return np.asarray(kept, dtype=int)


def calibrate_swinging_door(values: np.ndarray, target_retention: float) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if not 0 < target_retention <= 1:
        raise ValueError("target retention must be in (0,1]")
    span = float(np.ptp(x))
    if span <= 1e-15:
        epsilon = 0.0
        return epsilon, len(swinging_door_indices(x, epsilon)) / len(x)
    low, high = 0.0, span
    candidates: list[tuple[float, float]] = []
    for endpoint in (low, high):
        ratio = len(swinging_door_indices(x, endpoint)) / len(x)
        candidates.append((endpoint, ratio))
    for _ in range(40):
        mid = 0.5 * (low + high)
        ratio = len(swinging_door_indices(x, mid)) / len(x)
        candidates.append((mid, ratio))
        if ratio > target_retention:
            low = mid
        else:
            high = mid
    epsilon, ratio = min(candidates, key=lambda item: (abs(item[1] - target_retention), item[0]))
    return float(epsilon), float(ratio)


def reconstruct(values: np.ndarray, retained: np.ndarray, method: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    idx = np.asarray(retained, dtype=int)
    if len(idx) < 2 or idx[0] != 0 or idx[-1] != len(x) - 1 or np.any(np.diff(idx) <= 0):
        raise ValueError("retained indices must be strictly increasing and include both endpoints")
    grid = np.arange(len(x), dtype=float)
    if method == "LINEAR":
        return np.interp(grid, idx.astype(float), x[idx])
    if method == "ZOH":
        left = np.searchsorted(idx, np.arange(len(x)), side="right") - 1
        return x[idx[left]]
    raise ValueError(f"unknown reconstruction method: {method}")


def sampling_definition(mode: str, native_dt: float) -> tuple[str, float]:
    if mode.startswith("NATIVE_"):
        return "NATIVE", float(native_dt)
    if mode.startswith("ENDPOINT_"):
        kind = "ENDPOINT"
    elif mode.startswith("MEAN_"):
        kind = "MEAN"
    else:
        raise ValueError(f"unknown sampling mode: {mode}")
    token = mode.split("_", 1)[1]
    if token.endswith("MIN"):
        token = token[:-3]
    minutes = float(token.replace("P", "."))
    return kind, minutes


def sample_array(values: np.ndarray, mode: str, native_dt: float = 0.1) -> tuple[np.ndarray, float]:
    x = np.asarray(values, dtype=float)
    kind, output_dt = sampling_definition(mode, native_dt)
    if kind == "NATIVE":
        return x.copy(), float(native_dt)
    block = minutes_to_steps(output_dt, native_dt)
    n = (len(x) // block) * block
    if n < block:
        raise ValueError(f"series is too short for {output_dt:g}-minute sampling")
    shaped = x[:n].reshape(-1, block)
    if kind == "ENDPOINT":
        return shaped[:, -1].copy(), float(output_dt)
    return shaped.mean(axis=1), float(output_dt)


def sample_channels(
    channels: dict[str, np.ndarray], mode: str, native_dt: float = 0.1
) -> tuple[dict[str, np.ndarray], float]:
    sampled: dict[str, np.ndarray] = {}
    dt: float | None = None
    for key, values in channels.items():
        sampled_values, current_dt = sample_array(values, mode, native_dt=native_dt)
        if dt is not None and not math.isclose(dt, current_dt):
            raise AssertionError("inconsistent sampling interval")
        sampled[key] = sampled_values
        dt = current_dt
    if len({len(values) for values in sampled.values()}) != 1:
        raise AssertionError("sampled channel lengths differ")
    return sampled, float(dt)


def fidelity_metrics(observed: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    x = np.asarray(observed, dtype=float)
    y = np.asarray(reference, dtype=float)
    if len(x) != len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("invalid fidelity arrays")
    error = x - y
    rmse = float(np.sqrt(np.mean(error**2)))
    sd = float(np.std(y))
    span = float(np.ptp(y))
    corr = 1.0 if rmse <= 1e-15 else float(np.corrcoef(x, y)[0, 1])
    return {
        "rmse": rmse,
        "nrmse_sd": rmse / sd if sd > 1e-15 else math.nan,
        "nrmse_range": rmse / span if span > 1e-15 else math.nan,
        "correlation": corr,
    }


def chronological_split(n: int, minimum: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < minimum:
        raise ValueError(f"too few supervised examples: {n} < {minimum}")
    cut1 = n // 2
    cut2 = cut1 + n // 4
    return np.arange(cut1), np.arange(cut1, cut2), np.arange(cut2, n)


def scale_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(x_train), y_train)
    return np.asarray(model.predict(scaler.transform(x_eval)), dtype=float)


def make_lag_dataset(
    data: dict[str, np.ndarray],
    dt: float,
    *,
    horizon_minutes: float,
    baseline_lags_minutes: Sequence[float],
    candidate_lags_minutes: Sequence[float],
    context_keys: Sequence[str] = (),
    output_key: str = "measured_output",
    action_key: str = "action",
    disturbance_key: str = "disturbance",
    truth_key: str = "latent_output_truth",
) -> dict[str, Any]:
    horizon = minutes_to_steps(horizon_minutes, dt)
    base_lags = [minutes_to_steps(value, dt) for value in baseline_lags_minutes]
    candidate_lags = {float(value): minutes_to_steps(value, dt) for value in candidate_lags_minutes}
    start = max(max(base_lags), max(candidate_lags.values()))
    times = np.arange(start, len(data[output_key]) - horizon, dtype=int)
    if len(times) < 40:
        raise ValueError("insufficient lag-task support")
    columns = [
        np.asarray(data[channel])[times - lag]
        for channel in (output_key, action_key)
        for lag in base_lags
    ]
    columns.extend(np.asarray(data[key])[times] for key in context_keys)
    base = np.column_stack(columns)
    candidates = {
        minute: np.column_stack([base, np.asarray(data[disturbance_key])[times - lag]])
        for minute, lag in candidate_lags.items()
    }
    return {
        "base": base,
        "candidates": candidates,
        "target_measured": np.asarray(data[output_key])[times + horizon],
        "target_truth": np.asarray(data[truth_key])[times + horizon],
        "times": times,
    }


def evaluate_lag_task(
    data: dict[str, np.ndarray],
    dt: float,
    *,
    horizon_minutes: float,
    baseline_lags_minutes: Sequence[float],
    candidate_lags_minutes: Sequence[float],
    context_keys: Sequence[str] = (),
    task: str = "C1_LAG",
) -> dict[str, Any]:
    dataset = make_lag_dataset(
        data,
        dt,
        horizon_minutes=horizon_minutes,
        baseline_lags_minutes=baseline_lags_minutes,
        candidate_lags_minutes=candidate_lags_minutes,
        context_keys=context_keys,
    )
    y = dataset["target_measured"]
    y_truth = dataset["target_truth"]
    train, validation, test = chronological_split(len(y))
    pred_base_test = scale_fit_predict(dataset["base"][train], y[train], dataset["base"][test])
    validation_mse: dict[float, float] = {}
    candidate_test_predictions: dict[float, np.ndarray] = {}
    for minute, x in dataset["candidates"].items():
        pred_val = scale_fit_predict(x[train], y[train], x[validation])
        validation_mse[minute] = float(np.mean((y[validation] - pred_val) ** 2))
        candidate_test_predictions[minute] = scale_fit_predict(x[train], y[train], x[test])
    selected = min(validation_mse, key=lambda minute: (validation_mse[minute], minute))
    ordered_validation = sorted(validation_mse.items(), key=lambda item: (item[1], item[0]))
    best_validation_mse = float(ordered_validation[0][1])
    second_validation_mse = float(ordered_validation[1][1])
    pred_aug = candidate_test_predictions[selected]
    measured_base_mse = float(np.mean((y[test] - pred_base_test) ** 2))
    measured_aug_mse = float(np.mean((y[test] - pred_aug) ** 2))
    truth_base_mse = float(np.mean((y_truth[test] - pred_base_test) ** 2))
    truth_aug_mse = float(np.mean((y_truth[test] - pred_aug) ** 2))
    return {
        "task": task,
        "selected_lag_minutes": int(selected) if float(selected).is_integer() else float(selected),
        "validation_best_mse": best_validation_mse,
        "validation_second_mse": second_validation_mse,
        "validation_selection_margin": second_validation_mse - best_validation_mse,
        "validation_relative_selection_margin": (
            (second_validation_mse - best_validation_mse) / best_validation_mse
            if best_validation_mse > 1e-15
            else math.nan
        ),
        "measured_improvement": measured_base_mse - measured_aug_mse,
        "truth_improvement": truth_base_mse - truth_aug_mse,
        "improvement_positive": int(measured_base_mse - measured_aug_mse > 0.0),
        "truth_improvement_positive": int(truth_base_mse - truth_aug_mse > 0.0),
        "measured_baseline_mse": measured_base_mse,
        "measured_augmented_mse": measured_aug_mse,
        "truth_baseline_mse": truth_base_mse,
        "truth_augmented_mse": truth_aug_mse,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
    }


def spectral_features(
    values: np.ndarray,
    dt: float,
    *,
    window_minutes: float = 51.2,
    target_period: tuple[float, float] = (4.5, 6.2),
    background_periods: tuple[tuple[float, float], ...] = ((3.5, 4.3), (6.5, 8.0)),
) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    nperseg = int(round(window_minutes / dt))
    nperseg = min(nperseg, len(x))
    if nperseg < 16:
        raise ValueError("insufficient support for Welch spectrum")
    frequency, power = welch(
        x,
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        scaling="density",
    )
    positive = frequency > 0
    period = np.full_like(frequency, np.inf, dtype=float)
    period[positive] = 1.0 / frequency[positive]
    target = positive & (period >= target_period[0]) & (period <= target_period[1])
    background = np.zeros_like(positive, dtype=bool)
    for low, high in background_periods:
        background |= positive & (period >= low) & (period <= high)
    if not np.any(target) or not np.any(background):
        raise ValueError("target/background Fourier support is empty")
    target_indices = np.flatnonzero(target)
    peak_index = int(target_indices[np.argmax(power[target])])
    background_median = float(np.median(power[background]))
    if background_median <= 0:
        raise ValueError("non-positive spectral background")
    return {
        "prominence": float(power[peak_index] / background_median),
        "selected_period_minutes": float(period[peak_index]),
        "peak_power": float(power[peak_index]),
        "background_median": background_median,
        "nperseg": int(nperseg),
    }


def evaluate_spectral_tasks(
    data: dict[str, np.ndarray],
    dt: float,
    sampling: str,
    thresholds: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel, task in (("measured_output", "C2_OUTPUT"), ("disturbance", "C2_DISTURBANCE")):
        features = spectral_features(data[channel], dt)
        threshold = thresholds[(sampling, channel)]
        rows.append(
            {
                "task": task,
                **features,
                "threshold": threshold,
                "detected": int(features["prominence"] >= threshold),
            }
        )
    return rows


def make_history_dataset(
    data: dict[str, np.ndarray],
    dt: float,
    *,
    horizon_minutes: float,
    history_lags_minutes: Sequence[float],
    feature_keys: Sequence[str],
    output_key: str = "measured_output",
    truth_key: str = "latent_output_truth",
) -> dict[str, np.ndarray]:
    horizon = minutes_to_steps(horizon_minutes, dt)
    history_lags = [minutes_to_steps(value, dt) for value in history_lags_minutes]
    start = max(history_lags)
    times = np.arange(start, len(data[output_key]) - horizon, dtype=int)
    if len(times) < 40:
        raise ValueError("insufficient ranking-task support")
    features = np.column_stack(
        [np.asarray(data[channel])[times - lag] for channel in feature_keys for lag in history_lags]
    )
    return {
        "features": features,
        "current_measured": np.asarray(data[output_key])[times],
        "target_measured": np.asarray(data[output_key])[times + horizon],
        "target_truth": np.asarray(data[truth_key])[times + horizon],
    }


def rank_models(metrics: dict[str, float]) -> list[str]:
    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(MODEL_ORDER, key=lambda name: (metrics[name], order[name]))


def evaluate_model_ranking(
    data: dict[str, np.ndarray],
    dt: float,
    *,
    horizon_minutes: float,
    history_lags_minutes: Sequence[float],
    feature_keys: Sequence[str],
    task: str = "C3_RANKING",
) -> dict[str, Any]:
    dataset = make_history_dataset(
        data,
        dt,
        horizon_minutes=horizon_minutes,
        history_lags_minutes=history_lags_minutes,
        feature_keys=feature_keys,
    )
    x = dataset["features"]
    y = dataset["target_measured"]
    y_truth = dataset["target_truth"]
    train, validation, test = chronological_split(len(y))
    predictions: dict[str, np.ndarray] = {"PERSISTENCE": dataset["current_measured"][test]}
    predictions["RIDGE_AR"] = scale_fit_predict(x[train], y[train], x[test], alpha=1.0)
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=20260902,
    ).fit(x[train], y[train])
    predictions["SHALLOW_GBR"] = np.asarray(gbr.predict(x[test]), dtype=float)
    measured_mae = {name: float(np.mean(np.abs(y[test] - pred))) for name, pred in predictions.items()}
    truth_mae = {name: float(np.mean(np.abs(y_truth[test] - pred))) for name, pred in predictions.items()}
    measured_ranking = rank_models(measured_mae)
    truth_ranking = rank_models(truth_mae)
    measured_best = measured_mae[measured_ranking[0]]
    measured_second = measured_mae[measured_ranking[1]]
    truth_best = truth_mae[truth_ranking[0]]
    truth_second = truth_mae[truth_ranking[1]]
    result: dict[str, Any] = {
        "task": task,
        "measured_ranking": "|".join(measured_ranking),
        "truth_ranking": "|".join(truth_ranking),
        "best_model": measured_ranking[0],
        "truth_best_model": truth_ranking[0],
        "measured_selection_margin": measured_second - measured_best,
        "measured_relative_selection_margin": (
            (measured_second - measured_best) / measured_best if measured_best > 1e-15 else math.nan
        ),
        "truth_selection_margin": truth_second - truth_best,
        "truth_relative_selection_margin": (
            (truth_second - truth_best) / truth_best if truth_best > 1e-15 else math.nan
        ),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
    }
    for name in MODEL_ORDER:
        result[f"measured_mae_{name.lower()}"] = measured_mae[name]
        result[f"truth_mae_{name.lower()}"] = truth_mae[name]
    return result


def kendall_distance(ranking_a: str, ranking_b: str) -> float:
    a = ranking_a.split("|")
    b = ranking_b.split("|")
    if sorted(a) != sorted(b):
        raise ValueError("rankings contain different models")
    pos_a = {name: i for i, name in enumerate(a)}
    pos_b = {name: i for i, name in enumerate(b)}
    pairs = list(combinations(a, 2))
    discordant = sum((pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y]) < 0 for x, y in pairs)
    return float(discordant / len(pairs))


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
