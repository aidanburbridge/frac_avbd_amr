"""
Shared helpers for estimating simulation time steps from material properties,
voxel size, refinement level, and kinematic loading.

For the current AVBD fracture fixtures, a load-controlled estimate is usually
the more practical choice for quasi-static loading. The stricter wave-based
limit is still computed and returned for diagnostics and for more dynamic cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TimeStepEstimate:
    h_base: float
    h_effective: float
    max_ref_level: int
    effective_ref_level: int
    use_refined_size: bool
    density: float
    young_modulus: float
    poisson: float
    shear_modulus: float
    bulk_modulus: float
    shear_wave_speed: float
    dilatational_wave_speed: float
    wave_speed_kind: str
    wave_speed_used: float
    cfl_safety: float
    dt_wave: float
    load_speed: float
    load_safety: float
    delta_n0: float | None
    delta_s0: float | None
    dt_load: float | None
    policy: str
    chosen_limit: str
    recommended_dt: float

    def to_metadata(self) -> dict[str, float | int | bool | str | None]:
        return {
            "time_step_h_base": self.h_base,
            "time_step_h_effective": self.h_effective,
            "time_step_max_ref_level": self.max_ref_level,
            "time_step_effective_ref_level": self.effective_ref_level,
            "time_step_use_refined_size": self.use_refined_size,
            "time_step_density": self.density,
            "time_step_young_modulus": self.young_modulus,
            "time_step_poisson": self.poisson,
            "time_step_shear_modulus": self.shear_modulus,
            "time_step_bulk_modulus": self.bulk_modulus,
            "time_step_shear_wave_speed": self.shear_wave_speed,
            "time_step_dilatational_wave_speed": self.dilatational_wave_speed,
            "time_step_wave_speed_kind": self.wave_speed_kind,
            "time_step_wave_speed_used": self.wave_speed_used,
            "time_step_cfl_safety": self.cfl_safety,
            "time_step_dt_wave": self.dt_wave,
            "time_step_load_speed": self.load_speed,
            "time_step_load_safety": self.load_safety,
            "time_step_delta_n0": self.delta_n0,
            "time_step_delta_s0": self.delta_s0,
            "time_step_dt_load": self.dt_load,
            "time_step_policy": self.policy,
            "time_step_chosen_limit": self.chosen_limit,
            "time_step_recommended_dt": self.recommended_dt,
        }


def calc_shear_modulus(young_modulus: float, poisson: float) -> float:
    return float(young_modulus) / (2.0 * (1.0 + float(poisson)))


def calc_bulk_modulus(young_modulus: float, poisson: float) -> float:
    denom = 3.0 * (1.0 - 2.0 * float(poisson))
    if abs(denom) < 1e-12:
        raise ValueError("Poisson ratio is too close to 0.5 for a finite bulk modulus.")
    return float(young_modulus) / denom


def calc_shear_wave_speed(density: float, young_modulus: float, poisson: float) -> float:
    rho = float(density)
    if rho <= 0.0:
        raise ValueError("Density must be positive.")
    return float(np.sqrt(calc_shear_modulus(young_modulus, poisson) / rho))


def calc_dilatational_wave_speed(density: float, young_modulus: float, poisson: float) -> float:
    rho = float(density)
    if rho <= 0.0:
        raise ValueError("Density must be positive.")
    shear_mod = calc_shear_modulus(young_modulus, poisson)
    bulk_mod = calc_bulk_modulus(young_modulus, poisson)
    return float(np.sqrt((bulk_mod + (4.0 * shear_mod / 3.0)) / rho))


def calc_wave_dt(
    density: float,
    young_modulus: float,
    poisson: float,
    vox_size: float,
    *,
    wave_speed: str = "dilatational",
    safety: float = 1.0,
) -> float:
    h = float(vox_size)
    if h <= 0.0:
        raise ValueError("Voxel size must be positive.")

    mode = str(wave_speed).lower()
    if mode == "shear":
        c = calc_shear_wave_speed(density, young_modulus, poisson)
    elif mode == "dilatational":
        c = calc_dilatational_wave_speed(density, young_modulus, poisson)
    else:
        raise ValueError("wave_speed must be either 'shear' or 'dilatational'.")

    return float(safety) * h / c


def calc_damping(density: float, h: float, stiffness: float, zeta: float) -> float:
    """
    Preserve the current test-fixture damping convention.
    """
    mass = float(h) * float(h) * float(density)
    return 2.0 * float(zeta) * float(np.sqrt(mass * float(stiffness)))


def _load_speed(load_velocity: float | Sequence[float] | np.ndarray | None) -> float:
    if load_velocity is None:
        return 0.0
    arr = np.asarray(load_velocity, dtype=float)
    if arr.ndim == 0:
        return abs(float(arr))
    return float(np.linalg.norm(arr))


def estimate_timestep(
    *,
    density: float,
    young_modulus: float,
    poisson: float,
    h_base: float,
    max_ref_level: int = 0,
    load_velocity: float | Sequence[float] | np.ndarray | None = None,
    tensile_strength: float | None = None,
    use_refined_size: bool = True,
    policy: str = "min",
    wave_speed: str = "dilatational",
    cfl_safety: float = 0.3,
    load_safety: float = 0.05,
) -> TimeStepEstimate:
    h0 = float(h_base)
    if h0 <= 0.0:
        raise ValueError("Base voxel size must be positive.")

    max_level = max(0, int(max_ref_level))
    eff_level = max_level if use_refined_size else 0
    h_eff = h0 / (2 ** eff_level)

    rho = float(density)
    E = float(young_modulus)
    nu = float(poisson)

    shear_mod = calc_shear_modulus(E, nu)
    bulk_mod = calc_bulk_modulus(E, nu)
    shear_speed = calc_shear_wave_speed(rho, E, nu)
    dilatational_speed = calc_dilatational_wave_speed(rho, E, nu)

    wave_kind = str(wave_speed).lower()
    if wave_kind == "shear":
        wave_speed_used = shear_speed
    elif wave_kind == "dilatational":
        wave_speed_used = dilatational_speed
    else:
        raise ValueError("wave_speed must be either 'shear' or 'dilatational'.")

    dt_wave = calc_wave_dt(
        rho,
        E,
        nu,
        h_eff,
        wave_speed=wave_kind,
        safety=float(cfl_safety),
    )

    speed = _load_speed(load_velocity)
    delta_n0 = None
    delta_s0 = None
    dt_load = None
    if tensile_strength is not None:
        sigma_t = float(tensile_strength)
        delta_n0 = sigma_t * h_eff / E
        delta_s0 = sigma_t * h_eff / shear_mod
        if speed > 0.0:
            dt_load = float(load_safety) * min(delta_n0, delta_s0) / speed

    selected_policy = str(policy).lower()
    if selected_policy == "wave":
        chosen_limit = "wave"
        recommended_dt = dt_wave
    elif selected_policy == "load":
        if dt_load is None:
            chosen_limit = "wave"
            recommended_dt = dt_wave
        else:
            chosen_limit = "load"
            recommended_dt = dt_load
    elif selected_policy == "min":
        if dt_load is None or dt_wave <= dt_load:
            chosen_limit = "wave"
            recommended_dt = dt_wave
        else:
            chosen_limit = "load"
            recommended_dt = dt_load
    else:
        raise ValueError("policy must be one of 'wave', 'load', or 'min'.")

    return TimeStepEstimate(
        h_base=h0,
        h_effective=h_eff,
        max_ref_level=max_level,
        effective_ref_level=eff_level,
        use_refined_size=bool(use_refined_size),
        density=rho,
        young_modulus=E,
        poisson=nu,
        shear_modulus=shear_mod,
        bulk_modulus=bulk_mod,
        shear_wave_speed=shear_speed,
        dilatational_wave_speed=dilatational_speed,
        wave_speed_kind=wave_kind,
        wave_speed_used=wave_speed_used,
        cfl_safety=float(cfl_safety),
        dt_wave=dt_wave,
        load_speed=speed,
        load_safety=float(load_safety),
        delta_n0=delta_n0,
        delta_s0=delta_s0,
        dt_load=dt_load,
        policy=selected_policy,
        chosen_limit=chosen_limit,
        recommended_dt=float(recommended_dt),
    )
