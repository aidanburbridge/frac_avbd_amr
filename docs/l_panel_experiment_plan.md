# L-panel experiment plan

## What the recent runs suggest

### 1. Do not mix the old `run_033` / `run_034` results with the newer physical runs

`output/L_bar/run_033` and `output/L_bar/run_034` are useful as a historical baseline, but they are not directly comparable to the later `batch_1v2` / `run_038` style runs:

- they used the earlier geometry scaling state
- they used a much larger effective displacement scale
- they predate the explicit contact indenter metadata now present in the newer runs

That is why `run_034` reports first damage at about `4.54` displacement units and a final crack-area proxy of about `2792.77`, while the current physically scaled baseline reports first damage at about `0.0223 m` and final crack-area proxy of about `0.02335 m^2`.

### 2. The current best baseline is the physically scaled contact-indenter case

Use `output/batches/batch_1v2/L_bar_run_003` as the main reference point:

- `load_velocity = 0.03 m/s`
- `dt_physics = 2.5e-4 s`
- `steps = 10000`
- first damage onset: `0.7425 s`, `22.27 mm`
- peak crack-growth stage: `0.9735 s`, `29.20 mm`
- final broken bonds: `765`
- final crack-area proxy: `0.02335 m^2`
- final fracture work: `856.99 J`
- final kinetic energy: `1.23 J`
- solve time: `6878 s` (`~1.9 h`)

### 3. The displacement at damage onset is more stable than the time at damage onset

From the batch summary:

| Run | `v_load` | `dt` | Steps | First damage time | First damage disp. | Final broken bonds | Final crack area | Final fracture work | Final kinetic | Solve time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_003` | `0.03` | `2.5e-4` | `10000` | `0.7425 s` | `22.27 mm` | `765` | `0.02335 m^2` | `856.99 J` | `1.23 J` | `6878 s` |
| `run_004` | `0.03` | `1.25e-4` | `20000` | `0.7481 s` | `22.44 mm` | `589` | `0.01798 m^2` | `583.79 J` | `121.24 J` | `4691 s` |
| `run_005` | `0.02` | `2.5e-4` | `15000` | `1.1220 s` | `22.44 mm` | `748` | `0.02283 m^2` | `3782.53 J` | `32.32 J` | `12248 s` |

Main takeaways:

- Reducing `load_velocity` from `0.03` to `0.02 m/s` changed the event timing a lot, but the onset displacement barely changed.
- Halving `dt` reduced final damage and crack area for this setup.
- Residual kinetic energy changes a lot across runs, so it should be tracked as a numerical-quality flag, not just as a physical output.

For load-rate studies, compare curves against displacement, not against time.

## Parameter variations worth trying next

The cleanest way to proceed is in three passes: numerical sensitivity first, then boundary/loading sensitivity, then material sensitivity.

### A. Numerical sensitivity sweep

Goal: separate numerical artifacts from real fixture behavior.

| Parameter | Values to try | Why |
|---|---|---|
| `DT_PHYSICS` | `2.5e-4`, `1.67e-4`, `1.25e-4`, `1.0e-4` | Check time-step convergence and energy stability |
| `ITER` | `60`, `80`, `100`, `120` | Check solver convergence / contact stability |
| `VOX_RESOLUTION` | `160`, `200`, `240` | Check mesh-size sensitivity |
| `MAX_REF_LEVEL` | `1`, `2`, `3` | Check AMR sensitivity near the notch and crack path |
| `REFINE_STRESS_THRESHOLD` | `0.03`, `0.05`, `0.08` times `TENSILE_STRENGTH` | Check whether refinement is too eager or too late |

Recommended acceptance checks:

- first-damage displacement should stop moving much
- crack path and crack-area curve should stop changing materially
- final kinetic energy should stay low relative to fracture work
- `max_violation` and contact count history should not spike unexpectedly

### B. Loading and boundary-condition sweep

Goal: see how sensitive the benchmark is to how the force is introduced and supported.

| Parameter | Values to try | Why |
|---|---|---|
| `LOAD_VELOCITY[1]` | `0.01`, `0.015`, `0.02`, `0.03`, `0.04 m/s` | Rate sensitivity |
| `LOAD_PATCH_WIDTH` | `0.01`, `0.02`, `0.04`, `0.06 m` | Point-like vs distributed loading |
| `LOAD_OFFSET_FROM_RIGHT` | `0.02`, `0.03`, `0.04`, `0.05 m` | Move the loading patch relative to the corner |
| `LOAD_BAND_THICKNESS` | `0.01`, `0.02`, `0.04 m` | Change how deep the loading region is selected |
| `BOTTOM_FIX_DEPTH` | `0.01`, `0.02`, `0.04 m` | Support sensitivity |
| `LOAD_CONTACT_GAP_FACTOR` | `0.02`, `0.05`, `0.10` | Contact-init sensitivity |
| `LOAD_VOXEL_IDS` | current 2-voxel set vs 4-voxel vs 8-voxel surface patch | Directly test concentrated vs spread indenter loading |

Important control:

- keep the target final displacement the same across `LOAD_VELOCITY` runs by adjusting `steps = target_disp / (v_load * dt)`

Suggested target final displacement:

- `35 mm` to `40 mm`, since the current baseline already reaches peak growth around `29 mm`

### C. Material sensitivity sweep

Goal: separate fixture/setup effects from constitutive effects.

| Parameter | Values to try | Why |
|---|---|---|
| `E_MODULUS` | `1.5`, `2.0`, `2.5`, `3.0 GPa` | Global stiffness sensitivity |
| `TENSILE_STRENGTH` | `60`, `80`, `100 MPa` | Crack-initiation threshold sensitivity |
| `FRACTURE_TOUGHNESS` | `3e4`, `5e4`, `7e4 J/m^2-equivalent setting` | Post-initiation crack-growth sensitivity |
| `NU` | `0.25`, `0.30`, `0.35` | Lateral stress redistribution sensitivity |
| `PENALTY_GAIN` | `5e5`, `1e6`, `2e6` | Constraint stiffness / numerical response sensitivity |

If you only have time for one material sweep first:

- vary `TENSILE_STRENGTH` and `FRACTURE_TOUGHNESS` together

That pair will tell you more about crack-initiation vs crack-growth control than changing `E_MODULUS` alone.

## A practical sweep order

1. Fix the current physical baseline and run a small numerical sweep around it:
   - `dt`
   - `iterations`
   - `vox_resolution`
2. Once the baseline is numerically stable, sweep:
   - `load_velocity`
   - `load_patch_width`
   - `load_offset_from_right`
3. Then do the constitutive sweep:
   - `tensile_strength`
   - `fracture_toughness`
   - `youngs_modulus`

That order avoids spending time interpreting material trends that are really just time-step or loading-patch artifacts.

## Graphs that will show the differences clearly

### Best per-run overlays

Use `load_displacement_along_loading_axis` on the x-axis whenever you compare different loading rates.

Plot these curves for each run:

1. `crack_area_proxy` vs displacement
2. `broken_bond_count` vs displacement
3. `peak_mixed_mode_traction_utilization` vs displacement
4. `peak_stress_proxy` vs displacement
5. `kinetic`, `fracture_work`, `bond_potential`, `contact_potential` vs displacement
6. `max_violation` and `contact_count` vs displacement

Why this helps:

- velocity sweeps become directly comparable
- you can see whether different runs fail at the same displacement but different times
- you can separate physical response from numerical noise

### Best cross-run summary plots

Use the aggregate summary CSV for these:

1. Scatter plot:
   - x = `param_load_velocity`
   - y = `summary_final_crack_area_proxy`
   - color = `param_dt_physics`
   - bubble size = `meta_solve_time_s`

2. Scatter plot:
   - x = `param_dt_physics`
   - y = `summary_final_broken_bond_count`
   - color = `param_load_velocity`
   - bubble size = `summary_final_kinetic_energy`

3. Pareto plot:
   - x = `meta_solve_time_s`
   - y = `summary_final_crack_area_proxy`
   - label = run id

4. Heatmap for two-parameter sweeps:
   - axes = any two swept parameters
   - cell value = one response metric

Good heatmap response metrics:

- `summary_first_damage_onset_displacement`
- `summary_final_crack_area_proxy`
- `summary_final_broken_bond_count`
- `summary_final_kinetic_energy`
- `summary_peak_stress_proxy`

### Recommended “whole simulation” dashboard

If you want one page that summarizes each run well, use a 2x3 panel:

1. crack area vs displacement
2. broken bonds vs displacement
3. traction utilization vs displacement
4. peak stress proxy vs displacement
5. energy components vs displacement
6. table inset with:
   - first damage displacement
   - peak growth displacement
   - final crack area
   - final broken bonds
   - final kinetic energy
   - solve time

## The single strongest recommendation

For the next campaign, keep the current physically scaled contact-indenter setup and run:

- a `load_velocity` sweep
- a `dt` sweep
- a loading-patch-width sweep

Then compare everything on a displacement axis.

That combination should tell you quickly whether the observed response is mostly:

- rate-driven
- numerically driven
- or dominated by how the loading patch is being applied
