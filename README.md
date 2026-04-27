# smavbd

This repository's main solver is the Julia-backed solver in `jl_solver/`.
If you are deciding where to start, start there.

Most of the Python-side workflow routes into that backend through `jl_solver/hybrid_solver.py`, and `util/engine.py` defaults to `solver_type="hybrid"`.

## Where to look

- `jl_solver/`: main solver implementation and Julia bridge.
- `tests/L_bar.py`: representative fracture setup and a good starting fixture.
- `util/engine.py`: solver factory and headless execution loop.
- `util/batch_run.py`: batch runner for JSON manifests such as `batch_runs/example_batch_manifest.json`.

## Legacy / supporting code

- `py_solver/`: older Python solver kept as a fallback/reference.
- `AVBD_2D/`: older 2D prototype.
- `geometry/` and `util/`: voxelization, setup, export, plotting, and general helpers.

## Minimal example

```bash
python -m util.batch_run batch_runs/example_batch_manifest.json
```

This uses the default hybrid solver and writes outputs under `output/batches/`.
