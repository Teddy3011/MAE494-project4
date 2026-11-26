# Rubric audit (MAE 494/598 Project 4)

Evidence index against the official rubric. This is not a self-assigned score. Numbers are in the
generated files named here (`results/metrics.json` unless stated); none are repeated by hand.
Report = `docs/index.html` (generated from `src/site_template.html` by `run_all.py`).

## Structure and formulation (10)

| Requirement | Evidence | Test / artifact |
|---|---|---|
| Structure with >= 20 variables, nonlinear component, 3D presentation | Report section 2: SIMP voxel densities, one per non-passive voxel (3,648 on the 10 mm mesh) | `metrics.json` -> `designs.B.n` |
| Variables with units and bounds | Section 2 "Design variables" | `config.yaml` `simp.x_min` |
| Objective and constraints as explicit formulas | Section 2 "Optimization problem" (9 scaled constraints, g <= 0) | `src/bracket.py` `analyze` |
| Sourced limits or labelled assumptions | Section 2 source table; `config.yaml` comments | - |
| Mode descriptions | Section 8 results table (engine translation axis, engine share of modal mass) | `metrics.json` -> `designs.*.modes` |
| Coordinate system, loads, supports | Section 2 table; viewer markers | V5.eq*, V5.em*, V5.T |

Remaining weakness: all limits are project-brief assumptions; no external design standard sets them.

## Analysis code and verification (20)

| Requirement | Evidence | Test / artifact |
|---|---|---|
| Own FE solver (static, modal, buckling) | `src/fem.py`; section 3 | `tests/test_verification.py` |
| Ladder of >= 6 tests, value / exact / error table | Section 4 table (V1-V5 plus V6 in section 5) | `results/verification.csv` (all pass) |
| Exact solutions, convergence, global balance, invariance, rigid modes | V1 patch, V2 cantilever + locking, V3 frequency + convergence, V4 Euler column + convergence, V5 equilibrium, mass, rigid modes, symmetry, rotation | `results/figures/verification_convergence.png` |
| Bracket mesh convergence | Section 4 table | `results/mesh_convergence.csv` |
| Gradient checks | Section 5 | `results/gradient_check.csv` |

Remaining weaknesses:
- The first frequency is mesh dependent. The design is selected by a target sweep (`sweep` in `config.yaml`, `select()` in `run_all.py`) and meets 100 Hz on the 10 mm mesh and on the nested 5 mm mesh (`metrics.json` -> `checks.refined_B`), but gives below 100 Hz on a 6.67 mm mesh with both density transfers (`checks.mid_B`, `checks.mid_B_nearest`). The 1 Hz margin on the 5 mm mesh is smaller than this mesh-to-mesh spread; mesh convergence is not shown, and buckling was not recomputed at 5 mm.
- The solid-block buckling factor does not converge (singular prestress); reported in section 4.
- No comparison with commercial FE software (ANSYS results were never supplied).

## Sensitivity analysis (20)

| Requirement | Evidence | Test / artifact |
|---|---|---|
| Adjoint derivations (static, eigenvalue, buckling through state, filter chain) | Section 5 | `src/bracket.py` |
| Element-level AD + hand-assembled sparse adjoint (method ii) | `_stress_elem`, `_kg_elem` with `jax.grad`; `Bracket.analyze` | - |
| Gradient-check table, every function and variable location | Section 5 table | `results/gradient_check.csv`, `gradient_sweep.csv` |
| Taylor / step-size plot | Section 5 | `results/figures/gradient_taylor.png` |
| Independent method (i) check | Section 5 | `results/gradient_method_i.csv`; `test_gradients_method_i` |
| Cost vs finite differences | Section 5 cost table | `metrics.json` -> `gradient_cost` |

The complete central-difference gradient (all variables, all ten functions) is measured on the 20 mm mesh and compared component by component (section 5, "Complete finite-difference gradient"). Remaining weakness: on the 10 mm mesh the finite-difference cost is an estimate (2n x one measured analysis).

## Optimization (15)

| Requirement | Evidence | Test / artifact |
|---|---|---|
| Gradient-based algorithm, scaling | Section 6; `src/mma.py` | `test_mma_and_kkt_on_known_problem` |
| Continuation, starts | Section 6 tables (3 starts + no-buckling run on 20 mm) | `optimization_history.csv` |
| Final-stage settings changed one at a time | Section 6 "Final-stage settings we tried" | `results/convergence_study.csv` |
| Design-target sweep with refined-mesh selection rule | Section 6 sweep table | `metrics.json` -> `selection`; `test_selection_rule_needs_refined_margin` |
| Convergence history | Section 6 | `figures/optimization_history.png`, `constraint_history.png` |
| Active constraints, multipliers, interpretation | Section 6 KKT tables | `metrics.json` -> `runs.A/B.kkt` |
| Independent re-evaluation | Section 8 ("fresh model instance") | `src/optimize.py`, `src/report.py` |

Remaining weaknesses:
- KKT stationarity is about 1 % for A and B (`metrics.json` -> `runs.A/B.kkt`), with correct bound signs and near-zero complementarity; this is a well-converged local design, not a certified optimum. Topology problems have many local optima.
- If the last iterate is infeasible, the best feasible iterate of the last stage is reported instead (stated per run in section 6).

## Nonlinear component (15)

| Requirement | Evidence | Test / artifact |
|---|---|---|
| N1 linear buckling about the computed state | Sections 3 and 7 | V4, V5.kg |
| Sensitivity including K_G state dependence | Section 5 | `gradient_check.csv` (buckling rows), method (i) |
| Design with and without | Section 7, results table A vs B, viewer toggle | `figures/buckling_comparison.png` |

Remaining weakness: in the main problem buckling is inactive (zero multiplier), so A and B coincide. The
supplementary sweep without the frequency requirement shows the constraint becoming active, but its
masses are non-monotone (local optima), which weakens it as a cost-of-stability demonstration.

## Website and 3D comparison (10)

| Requirement (course viewer list) | Evidence |
|---|---|
| Initial vs optimized, linked cameras | `docs/assets/viewer.js` `Pane`, controls `change` sync |
| True geometry, exaggeration stated | deformation slider label (x0 = true geometry) |
| Colour by key constraint field with colour bar, no rainbow, violations not colour-only | cividis utilization, colour bar with limit tick, red outline boxes for voxels over the limit |
| State shown with shape-function interpolation | node displacements per load case on voxel faces |
| Explanatory animation | vibration mode 1 and buckling mode 1 |
| Hover / tap inspection | element id, density, utilization, displacement |
| Metrics panel with pass/fail | tables under each pane (from `results.json`) |
| Loads from results.json, phone width, fast render, error message | checked on a local server: no console errors, no horizontal scroll at 375 px, explicit 404 message |

Remaining weaknesses:
- The live site (https://teddy3011.github.io/MAE494-project4/) still shows the graded `6c272b2` version. The `student-polish` branch is local and unpushed; its page was checked on a local server (desktop and 375 px, no console errors, figures served, both animations, 114 MathJax formulas). Not checked in a private browser window by the team.

## Reproducibility (5)

| Requirement | Evidence |
|---|---|
| One command regenerates everything | `python run_all.py` (exits non-zero on any failed check); last final run time in `results/metrics.json` -> `pipeline_seconds` |
| Pinned dependencies, seeds, environment | `requirements.txt` (pip freeze), `config.yaml` `seed`, `results/environment.json` |
| README with exact steps | `README.md` |

Remaining weakness: tested on one Windows machine only. The full `run_all.py` (final mode, about 107 min) was run in the `student-polish` worktree; the fresh-clone install and `pytest` check dates from commit `4cfd53a`. The convergence study (`studies/convergence_study.py`, about 50 min) is separate from `run_all.py`.

## Presentation and clarity (5)

The report follows the ten required sections and was rewritten in a student voice, with a "What we learned" and a
"Things that did not work" section. The in-class presentation itself is outside this repository.
Open items for the team: names, contributions and review assignments are TODO placeholders in the report
("Team and contributions") and in `STUDENT_REVIEW_CHECKLIST.md`. The sentence "The student team reviewed the
sections identified in the contribution statement." should be added to the AI-use statement only after those reviews are done.
