"""
Unified timestep selection for AVBD-style voxel benchmarks.

This repository uses an implicit AVBD-style solver rather than a strict explicit
CFL scheme. The public timestep rule is nevertheless chosen from a single
material/loading estimate:

    dt = min(dt_load, dt_wave)

The load-based term controls quasi-static benchmarks, while the wave-based term
acts as a dynamic cap and is especially important for projectile-impact cases.

The final solver does not use diagonal bonds, so the public load-controlled
scale is based only on the normal bond opening length scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ETA_LOAD = 0.05
ETA_WAVE = 0.20


@dataclass(frozen=True)
class TimeStepEstimate:
    h_base: float
    h_min: float
    max_ref_level: int
    density: float
    young_modulus: float
    poisson: float
    shear_modulus: float
    bulk_modulus: float
    dilatational_wave_speed: float
    dt_wave: float
    load_speed: float
    delta0: float | None
    dt_load: float | None
    recommended_dt: float

    def to_metadata(self) -> dict[str, float | int | None]:
        return {
            "time_step_h_base": self.h_base,
            "time_step_h_min": self.h_min,
            "time_step_max_ref_level": self.max_ref_level,
            "time_step_density": self.density,
            "time_step_young_modulus": self.young_modulus,
            "time_step_poisson": self.poisson,
            "time_step_shear_modulus": self.shear_modulus,
            "time_step_bulk_modulus": self.bulk_modulus,
            "time_step_dilatational_wave_speed": self.dilatational_wave_speed,
            "time_step_dt_wave": self.dt_wave,
            "time_step_load_speed": self.load_speed,
            "time_step_delta0": self.delta0,
            "time_step_dt_load": self.dt_load,
            "time_step_recommended_dt": self.recommended_dt,
        }


def calc_shear_modulus(young_modulus: float, poisson: float) -> float:
    return float(young_modulus) / (2.0 * (1.0 + float(poisson)))


def calc_bulk_modulus(young_modulus: float, poisson: float) -> float:
    denom = 3.0 * (1.0 - 2.0 * float(poisson))
    if abs(denom) < 1e-12:
        raise ValueError("Poisson ratio is too close to 0.5 for a finite bulk modulus.")
    return float(young_modulus) / denom


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
    h_min: float,
) -> float:
    h = float(h_min)
    if h <= 0.0:
        raise ValueError("Minimum voxel size must be positive.")

    c_dil = calc_dilatational_wave_speed(density, young_modulus, poisson)
    return ETA_WAVE * h / c_dil


def calc_damping(density: float, h: float, stiffness: float, zeta: float) -> float:
    """
    Preserve the current test-fixture damping convention.
    """
    mass = float(h) * float(h) * float(density)
    return 2.0 * float(zeta) * float(np.sqrt(mass * float(stiffness)))


def calc_render_dt(dt_physics: float, steps_per_export: int) -> float:
    dt = float(dt_physics)
    if dt <= 0.0:
        raise ValueError("Physics time step must be positive.")
    stride = max(1, int(steps_per_export))
    return dt * stride


def calc_num_render_frames(num_physics_steps: int, steps_per_export: int) -> int:
    steps = int(num_physics_steps)
    if steps <= 0:
        raise ValueError("Number of physics steps must be positive.")
    stride = max(1, int(steps_per_export))
    return 1 + ((steps + stride - 1) // stride)


def calc_target_duration(
    target_displacement: float,
    load_velocity: float | Sequence[float] | np.ndarray,
) -> float:
    displacement = float(target_displacement)
    if displacement <= 0.0:
        raise ValueError("Target displacement must be positive.")

    speed = _load_speed(load_velocity)
    if speed <= 0.0:
        raise ValueError("Load velocity must define a positive characteristic speed.")

    return displacement / speed


def calc_num_physics_steps_for_target_displacement(
    target_displacement: float,
    dt_physics: float,
    load_velocity: float | Sequence[float] | np.ndarray,
) -> int:
    dt = float(dt_physics)
    if dt <= 0.0:
        raise ValueError("Physics time step must be positive.")

    duration = calc_target_duration(target_displacement, load_velocity)
    return max(1, int(np.ceil(duration / dt)))


def print_timestep_schedule(
    dt_physics: float,
    steps_per_export: int,
    num_physics_steps: int,
) -> None:
    render_dt = calc_render_dt(dt_physics, steps_per_export)
    num_frames = calc_num_render_frames(num_physics_steps, steps_per_export)
    print(
        f"Timesteps: physics_dt={float(dt_physics):.6e} s, "
        f"render_dt={render_dt:.6e} s, "
        f"physics_steps={int(num_physics_steps)}, "
        f"render_frames={num_frames}"
    )


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
) -> TimeStepEstimate:
    """
    Estimate a physics timestep from a unified material/loading rule.

    The solver is implicit AVBD-style, so this is not described as a strict
    explicit-CFL stability law. The load-based term is the main control for
    quasi-static loading, while the wave-based term provides a dynamic cap.
    """
    h0 = float(h_base)
    if h0 <= 0.0:
        raise ValueError("Base voxel size must be positive.")

    max_level = max(0, int(max_ref_level))
    h_min = h0 / (2 ** max_level)

    rho = float(density)
    E = float(young_modulus)
    nu = float(poisson)
    if rho <= 0.0:
        raise ValueError("Density must be positive.")
    if E <= 0.0:
        raise ValueError("Young's modulus must be positive.")

    shear_mod = calc_shear_modulus(E, nu)
    bulk_mod = calc_bulk_modulus(E, nu)
    dilatational_speed = calc_dilatational_wave_speed(rho, E, nu)
    dt_wave = calc_wave_dt(rho, E, nu, h_min)

    speed = _load_speed(load_velocity)
    delta0 = None
    dt_load = None
    if tensile_strength is not None:
        sigma_t = float(tensile_strength)
        if sigma_t > 0.0:
            delta0 = sigma_t * h_min / E
            if speed > 0.0:
                dt_load = ETA_LOAD * delta0 / speed

    recommended_dt = dt_wave if dt_load is None else min(dt_load, dt_wave)

    return TimeStepEstimate(
        h_base=h0,
        h_min=h_min,
        max_ref_level=max_level,
        density=rho,
        young_modulus=E,
        poisson=nu,
        shear_modulus=shear_mod,
        bulk_modulus=bulk_mod,
        dilatational_wave_speed=dilatational_speed,
        dt_wave=dt_wave,
        load_speed=speed,
        delta0=delta0,
        dt_load=dt_load,
        recommended_dt=float(recommended_dt),
    )
