# Engine-mount bracket: PDE-constrained topology optimization

MAE 494/598 Design Optimization, Project 4 (ASU). Minimum-mass aluminium 6061-T6 engine-mount
bracket under stress, mount-displacement, natural-frequency and linear-buckling constraints.
The 3D hex8 finite-element solver, all adjoint sensitivities and the optimizer (MMA) are our own code
(NumPy/SciPy sparse for global algebra, JAX for element-level automatic differentiation), generated
and revised with heavy use of Claude Code and checked with the verification suite. See the AI-use
statement in the report and `STUDENT_REVIEW_CHECKLIST.md`.

Report and interactive 3D comparison: `docs/index.html`, served by GitHub Pages from `main` / `docs`
(https://teddy3011.github.io/MAE494-project4/).

## Reproduce

Tested with Python 3.11 on Windows 11. Every dependency is pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest -q                            # 15 tests, about 1 min
.venv/Scripts/python run_all.py                              # everything, final mode
```

`run_all.py` regenerates, from source, every file in `results/` and the generated parts of `docs/`
(`index.html`, `assets/results.json`, `assets/figures/`). It exits non-zero if any verification row or
gradient check fails. Final mode runs in two phases of parallel processes:

1. a base design without the buckling constraint on the 10 mm mesh (full continuation from the
   solid block), together with nine 20 mm studies (starting designs, buckling side study);
2. a design-target sweep (`sweep` in `config.yaml`): from the base design, the final stage at each
   frequency target with and without buckling, each checked on a 5 mm mesh. `select()` in
   `run_all.py` picks the lightest with-buckling design that passes the 5 mm check with the
   required margin.

On a 16-thread laptop (see `results/environment.json`) the last final run took about 107 min
(the exact time is in the report, section 10). Every optimization run is saved in `results/run_cache/`
as it finishes; `python run_all.py --resume` reuses them after an interruption. `python run_all.py --quick` checks the pipeline in a few minutes (all on 20 mm, shortened
continuation; the page is then stamped "QUICK MODE" and is not for submission).
`python run_all.py --report-only` rebuilds figures and the page from `results/runs.pkl`.

The final-stage convergence study is separate (about 50 min):
`python studies/convergence_study.py` writes `results/convergence_study.csv`, which the report reads.
It starts from the committed design `studies/start_B107_4cfd53a.npy`.

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
| `studies/` | final-stage convergence study and its starting design |
| `STUDENT_REVIEW_CHECKLIST.md` | what each team member should be able to explain before submission |
| `results/` | generated: `verification.csv`, `gradient_*.csv`, `mesh_convergence.csv`, `optimization_history.csv`, `convergence_study.csv`, `metrics.json`, `results.json`, `figures/`, `environment.json`, `rubric_audit.md` |

Generated files are overwritten by `run_all.py`; do not edit them by hand.
