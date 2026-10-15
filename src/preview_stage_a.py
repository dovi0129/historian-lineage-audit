"""CSTR trajectory generation from the original preview geometry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import cstr_dynamic as cd
from osc_disturbance import DampedResonatorDisturbance

CSTR_AR_STREAM = 91801
CSTR_OSC_STREAM = 91802
CSTR_OBS_STREAM = 91803
CSTR_PREVIEW_STREAM = 91804


def round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def dimensionless_horizon_grid(l_dyn50: int) -> list[int]:
    """Map {0,.5,1,1.5,2} x L to unique non-negative integer steps."""
    if l_dyn50 < 1:
        raise ValueError(f"l_dyn50 must be positive, got {l_dyn50}")
    return sorted({round_half_up(r * l_dyn50) for r in (0.0, 0.5, 1.0, 1.5, 2.0)})


def _future_prefix(values: np.ndarray, start: int, width: int) -> np.ndarray:
    out = np.zeros(int(width), dtype=float)
    if start < len(values):
        n = min(width, len(values) - start)
        out[:n] = values[start : start + n]
    return out


def ar_trace(
    n: int,
    rho: float,
    innovation_sd: float,
    seed: int,
    stream: int,
    *,
    stationary_start: bool = False,
) -> np.ndarray:
    """AR(1) trace with an isolated deterministic random stream."""
    rng = np.random.default_rng([int(seed), int(stream)])
    out = np.zeros(int(n), dtype=float)
    if stationary_start:
        out[0] = rng.normal(0.0, innovation_sd / np.sqrt(max(1.0 - rho**2, 1e-12)))
    for t in range(1, int(n)):
        out[t] = rho * out[t - 1] + rng.normal(0.0, innovation_sd)
    return out


def oscillator_trace(
    n: int,
    *,
    rho: float,
    sigma: float,
    period_steps: float,
    seed: int,
    stream_offset: int = 0,
    amp_scale: float = 1.0,
    k_amp: float = 0.0,
    r_target: float | None = None,
) -> np.ndarray:
    """Generate a damped-resonator trace without consuming plant RNG state."""
    osc = DampedResonatorDisturbance(
        rho=rho,
        sigma=sigma,
        omega=2.0 * np.pi / period_steps,
        k_amp=k_amp,
        r_target=r_target,
        amp_scale=amp_scale,
        seed=int(seed) + int(stream_offset),
    )
    return osc.simulate(int(n))


def forecast_error_matrix(n: int, width: int, sd: float, seed: int, stream: int) -> np.ndarray:
    rng = np.random.default_rng([int(seed), int(stream)])
    return rng.normal(0.0, float(sd), size=(int(n), int(width)))

@dataclass(frozen=True)
class CSTRTraceConfig:
    n_steps: int = 500
    setpoint: float = 350.0
    bg_rho: float = 0.90
    bg_stationary_sd: float = 0.20
    osc_stationary_sd: float = 0.35
    osc_rho_per_minute: float = 0.94
    period_minutes: float = 5.33
    obs_sd: float = 0.05
    preview_error_rel_sd: float = 0.5


def make_cstr_traces(cfg: CSTRTraceConfig, seed: int, max_preview: int, osc_scale: float = 1.0):
    n = int(cfg.n_steps)
    bg_innov = cfg.bg_stationary_sd * np.sqrt(max(1.0 - cfg.bg_rho**2, 1e-12))
    bg = ar_trace(n, cfg.bg_rho, bg_innov, seed, CSTR_AR_STREAM, stationary_start=True)

    osc_rho = cfg.osc_rho_per_minute ** cd.DT
    osc_sigma = cfg.osc_stationary_sd * np.sqrt(max(1.0 - osc_rho**2, 1e-12))
    osc = oscillator_trace(
        n,
        rho=osc_rho,
        sigma=osc_sigma,
        period_steps=cfg.period_minutes / cd.DT,
        seed=seed,
        stream_offset=CSTR_OSC_STREAM,
        amp_scale=osc_scale,
    )
    obs_rng = np.random.default_rng([int(seed), CSTR_OBS_STREAM])
    obs = obs_rng.normal(0.0, cfg.obs_sd, size=n)
    err = forecast_error_matrix(
        n,
        max_preview,
        cfg.preview_error_rel_sd * abs(float(osc_scale)) * cfg.osc_stationary_sd,
        seed,
        CSTR_PREVIEW_STREAM,
    )
    return {"ar": bg, "osc": osc, "obs": obs, "preview_error": err}


class CSTRPreviewNMPC(cd.NMPC):
    """Nominal CSTR NMPC with a future disturbance-rate prefix."""

    def __init__(self, *args: Any, preview_h: int, preview_noise_scale: float, **kwargs: Any) -> None:
        kwargs["offset_free"] = False
        super().__init__(*args, **kwargs)
        self.preview_h = int(preview_h)
        self.preview_noise_scale = float(preview_noise_scale)
        self._preview_rate = np.zeros(self.H, dtype=float)

    def _predict(self, ca, temp, tc, calib):
        traj = []
        for i in range(self.H):
            rate_dist = float(self._preview_rate[i])

            def deriv(x, t, u, disturbance=rate_dist):
                dca, dt = cd.deriv_model(x, t, u, calib)
                return dca, dt + disturbance

            ca, temp = cd.advance(ca, temp, tc, deriv, dt=self.dt)
            traj.append(temp)
        return traj

    def act(self, temp, temp_sp, *, dist_preview=None, preview_error=None, **kwargs):
        self._preview_rate[:] = 0.0
        exact = np.asarray(dist_preview if dist_preview is not None else [], dtype=float)
        error = np.asarray(preview_error if preview_error is not None else [], dtype=float)
        n = min(self.preview_h, self.H, len(exact))
        if n:
            self._preview_rate[:n] = exact[:n] + self.preview_noise_scale * error[:n]
        return super().act(temp, temp_sp, **kwargs)


def run_cstr_preview(
    *,
    cfg: CSTRTraceConfig,
    seed: int,
    max_preview: int,
    controller_h: int,
    preview_h: int,
    preview_noise_scale: float,
    osc_scale: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    traces = make_cstr_traces(cfg, seed, max_preview, osc_scale=osc_scale)
    calib = dict(cd.NOMINAL)
    controller = CSTRPreviewNMPC(
        calib,
        H=controller_h,
        rho=0.05,
        preview_h=preview_h,
        preview_noise_scale=preview_noise_scale,
    )
    tc0 = cd.tc_for_setpoint(cfg.setpoint, lambda tc: cd.steady_model(tc, calib))
    ca, temp = cd.steady_model(tc0, calib)
    controller.reset(tc0)

    temps: list[float] = []
    actions: list[float] = []
    delta_actions: list[float] = []
    prev_tc = float(tc0)
    n = int(cfg.n_steps)
    for t in range(n):
        exact = _future_prefix(traces["osc"], t, max_preview)
        err = traces["preview_error"][t]
        temp_obs = float(temp + traces["obs"][t])
        tc = controller.act(
            temp_obs,
            cfg.setpoint,
            r_hat=0.0,
            dist_preview=exact,
            preview_error=err,
        )
        disturbance = float(traces["ar"][t] + traces["osc"][t])

        def deriv(x, tt, u, d=disturbance):
            dca, dtemp = cd.deriv_model(x, tt, u, calib)
            return dca, dtemp + d

        ca, temp = cd.advance(ca, temp, tc, deriv)
        temps.append(float(temp))
        actions.append(float(tc))
        delta_actions.append(float(tc - prev_tc))
        prev_tc = float(tc)

    temp_arr = np.asarray(temps)
    action_arr = np.asarray(actions)
    delta_arr = np.asarray(delta_actions)
    dev = temp_arr - cfg.setpoint
    step_cost = dev**2 + controller.rho * delta_arr**2
    sat = (np.isclose(action_arr, cd.TC_LO, atol=1e-8) | np.isclose(action_arr, cd.TC_HI, atol=1e-8))
    rate_sat = np.isclose(np.abs(delta_arr), cd.TC_RATE, atol=1e-8)
    violation = (
        (action_arr < cd.TC_LO - 1e-8)
        | (action_arr > cd.TC_HI + 1e-8)
        | (np.abs(delta_arr) > cd.TC_RATE + 1e-8)
    )
    metrics = {
        "cost": float(np.mean(step_cost)),
        "rmse": float(np.sqrt(np.mean(dev**2))),
        "in_band": float(np.mean(np.abs(dev) <= 0.30)),
        "tv": float(np.sum(np.abs(delta_arr))),
        "sat_frac": float(np.mean(sat | rate_sat)),
        "constraint_violation_frac": float(np.mean(violation)),
        "n_steps": int(n),
    }
    trajectory = {"output": temp_arr, "action": action_arr, "disturbance": traces["ar"] + traces["osc"], **traces}
    return metrics, trajectory


def first_fraction_response(delta: np.ndarray, terminal: float, fraction: float = 0.5) -> int:
    """First step reaching a fraction of an independently computed steady response."""
    delta = np.asarray(delta, dtype=float)
    target = float(fraction) * abs(terminal)
    if not np.isfinite(target) or target <= 1e-12:
        raise ValueError("terminal response is too small to define a dynamic scale")
    hit = np.flatnonzero(np.abs(delta) >= target)
    if not len(hit):
        raise ValueError("response did not reach requested fraction")
    return int(hit[0] + 1)


def cstr_action_response_l50(steps: int = 80, setpoint: float = 350.0) -> int:
    calib = dict(cd.NOMINAL)
    tc0 = cd.tc_for_setpoint(setpoint, lambda tc: cd.steady_model(tc, calib))
    ca0, t0 = cd.steady_model(tc0, calib)
    base_ca, base_t = ca0, t0
    step_ca, step_t = ca0, t0
    tc1 = float(np.clip(tc0 + 1.0, cd.TC_LO, cd.TC_HI))
    base, stepped = [], []
    for _ in range(int(steps)):
        base_ca, base_t = cd.advance(base_ca, base_t, tc0, lambda ca, t, tc: cd.deriv_model(ca, t, tc, calib))
        step_ca, step_t = cd.advance(step_ca, step_t, tc1, lambda ca, t, tc: cd.deriv_model(ca, t, tc, calib))
        base.append(base_t)
        stepped.append(step_t)
    terminal = cd.steady_model(tc1, calib)[1] - cd.steady_model(tc0, calib)[1]
    return first_fraction_response(np.asarray(stepped) - np.asarray(base), terminal)
