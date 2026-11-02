# Engine-mount bracket: PDE-constrained design optimization

MAE 494/598 Design Optimization, Project 4 (ASU). A 3D linear-elastic hex8 finite-element
solver written from scratch (NumPy/SciPy sparse, JAX for element-level AD), used to minimize
the mass of an aluminium 6061-T6 engine-mount bracket under stress, displacement,
frequency and linear-buckling constraints.

Report site: `docs/` (GitHub Pages).

## Reproduce

Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest -q                          # full suite, < 2 min
.venv/Scripts/python run_all.py                            # regenerates everything in results/
```

All assumptions (material, loads, limits, solver options) are in `config.yaml`. Random seed: `seed` there.

## Layout

| Path | Contents |
|---|---|
| `src/fem.py` | hex8 element (JAX), sparse assembly, static solve |
| `src/verification.py` | verification ladder; writes `results/verification.csv` |
| `tests/` | pytest suite |
| `results/` | generated outputs, `environment.json` |
| `docs/` | GitHub Pages site |

## Status

| Stage | Status |
|---|---|
| 0 Scaffold | done (GitHub repo and Pages pending owner/repo) |
| 1 Static solver, V1 patch, V2 cantilever | done |
| 2-8 | not started |
