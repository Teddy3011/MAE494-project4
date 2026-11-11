# Engine-mount bracket: PDE-constrained topology optimization

MAE 494/598 Design Optimization, Project 4 (ASU). Minimum-mass aluminium 6061-T6 engine-mount
bracket under stress, mount-displacement, natural-frequency and linear-buckling constraints.
The 3D hex8 finite-element solver, all adjoint sensitivities and the optimizer (MMA) are written
from scratch: NumPy/SciPy sparse for global algebra, JAX for element-level automatic differentiation.

Report and interactive 3D comparison: `docs/index.html`, served by GitHub Pages from `main` / `docs`
(https://teddy3011.github.io/MAE494-project4/ once Pages is enabled).

## Reproduce

Tested with Python 3.11 on Windows 11. Every dependency is pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest -q                            # 12 tests, about 35 s
.venv/Scripts/python run_all.py                              # everything, final mode
```

`run_all.py` regenerates, from source, every file in `results/` and the generated parts of `docs/`
(`index.html`, `assets/results.json`, `assets/figures/`). It exits non-zero if any verification row or
gradient check fails. Final mode runs the two principal optimizations on the 10 mm mesh and nine
20 mm studies in parallel processes; on a 16-thread laptop (see `results/environment.json`) it takes
about an hour. `python run_all.py --quick` checks the pipeline in a few minutes (all on 20 mm, shortened
continuation; the page is then stamped "QUICK MODE" and is not for submission).
`python run_all.py --report-only` rebuilds figures and the page from `results/runs.pkl`.

To view the site locally: `python -m http.server -d docs` and open http://localhost:8000.

All assumptions (material, domain, loads, limits, SIMP and optimizer settings, seed) are in `config.yaml`.

## Layout

| Path | Contents |
|---|---|
| `src/fem.py` | hex8 element with incompatible modes (JAX): stiffness, mass, Gauss-point stress, geometric stiffness; sparse assembly, static, modal and buckling solvers |
| `src/bracket.py` | voxel SIMP bracket model, filter, projection, load cases, all responses and their adjoint gradients |
| `src/mma.py`, `src/optimize.py` | Method of Moving Asymptotes, continuation driver, history, KKT check |
| `src/verification.py`, `src/gradcheck.py` | verification ladder V1-V5; gradient checks V6 (central differences, Taylor, dense JAX method (i), cost) |
| `src/report.py`, `src/site.py`, `src/site_template.html` | figures, `metrics.json`, `results.json`, generated report page |
| `docs/assets/viewer.js`, `docs/assets/style.css` | three.js viewer and page style (hand-written) |
| `tests/` | pytest suite |
| `results/` | generated: `verification.csv`, `gradient_*.csv`, `mesh_convergence.csv`, `optimization_history.csv`, `metrics.json`, `results.json`, `figures/`, `environment.json`, `rubric_audit.md` |

Generated files are overwritten by `run_all.py`; do not edit them by hand.
