# Student review checklist

Every team member should be able to explain each item below in their own words, without the
report open. Each item says where to look. Tick an item only after you can explain it to a
teammate and answer the check question.

Most of the code and report text was generated with Claude Code (see the AI-use statement in
the report). This checklist is how we make sure we understand and stand behind it.

| Member | Items reviewed | Date |
|---|---|---|
| TODO: name | TODO | TODO |
| TODO: name | TODO | TODO |
| TODO: name | TODO | TODO |
| TODO: name | TODO | TODO |

## 1. The three load cases
- [ ] LC1: 3 g vertical, W = 3 x 120 x 9.81 = 3531.6 N downward, spread over the 80 x 40 mm top patch.
- [ ] LC2 and LC3: LC1 plus and minus a 400 N·m torque about y. Why both signs?
- Where: `config.yaml` (`loads`), `src/bracket.py` (`Bracket.__init__`, the `F` array), report section 2.
- Check: why is there no buckling constraint for LC1? (Its prestress is the mean of LC2 and LC3, so its buckling factor is at least the smaller of theirs.)

## 2. How the torque gives zero net force and 400 N·m
- [ ] Each interface node gets f_i = -T (x_i - x_c) w_i / sum_j w_j (x_j - x_c)^2.
- [ ] Net force: the weights are symmetric about x_c, so sum w_i (x_i - x_c) = 0.
- [ ] Moment about y through the patch centre: -sum (x_i - x_c) f_i = T.
- Where: `src/bracket.py`, verification rows V5.T and V5.em*, test `test_mesh_transfer_preserves_model`.
- Check: what would change if the patch were not symmetric about x_c?

## 3. SIMP, filtering and projection
- [ ] SIMP: E = E0 (eps + (1 - eps) rho^p). Why does p = 3 push densities to 0 or 1?
- [ ] Density filter, radius 30 mm: minimum member size and no checkerboards. Why a radius in mm, not in elements?
- [ ] Heaviside projection with beta: makes the field crisper. What happens to the gradient as beta grows?
- [ ] Why the continuation starts at p = 2, not p = 1.
- [ ] Why the modal mass and the buckling prestress use different interpolations from the stiffness.
- Where: `src/bracket.py` (module docstring, `densities`, `analyze`), report section 3.

## 4. Incompatible modes and shear locking
- [ ] Why a plain 8-node hex is too stiff in bending (V2.b: 33 % too stiff).
- [ ] What the 9 bubble modes add, and why they are condensed out at element level.
- [ ] Why Taylor's correction is needed for the patch test (V1).
- Where: `src/fem.py` (`hex8_B`), verification rows V1 and V2.

## 5. One static adjoint derivation (be able to do it on the board)
- [ ] From K u = F: du = -K^-1 (dK) u.
- [ ] Solve K lambda = -(dg/du)^T once, then dg/dx_e = dg/dx_e (explicit) + lambda_e^T (dK_e/dx_e) u_e.
- [ ] Why this costs one solve per constraint, not one per design variable.
- Where: `src/bracket.py` (`analyze`, "static functionals" block), report section 5, gradient tables.

## 6. The modal sensitivity
- [ ] d(lambda_i) = phi_i^T (dK - lambda_i dM) phi_i for mass-normalized phi_i. Why no extra solve?
- [ ] Why we use a KS smooth minimum over six modes (repeated or crossing modes).
- [ ] Why the constraint is written as 1 - KSmin(lambda_i / lambda_req), not lambda_req / lambda_1 - 1.
- Where: `src/bracket.py` ("natural frequencies" block), report sections 2 and 5.

## 7. Why buckling needs a prestress sensitivity
- [ ] K_G is built from the static stress, and that stress changes with density.
- [ ] The extra term -c^T du/dx needs one more adjoint solve: K lambda_b = alpha_req sum_i w_i c_i.
- [ ] What would go wrong if K_G were treated as fixed.
- Where: `src/bracket.py` ("linear buckling" block), `src/gradcheck.py` (method (i) check), report section 5.

## 8. Why the optimization target is above 100 Hz
- [ ] The 10 mm voxels overestimate stiffness. Our first 100 Hz design gave 94.8 Hz on a 5 mm mesh (commit 8189727).
- [ ] How the target sweep and the selection rule in `run_all.py` (`select`) pick the design.
- [ ] Why we keep a margin on the 5 mm mesh instead of taking a design at exactly 100 Hz.
- Where: `config.yaml` (`sweep`), report sections 2, 6 and 8.

## 9. Why frequency controls the design
- [ ] Rough check: 120 kg at 100 Hz needs k = m (2 pi f)^2, about 47 kN/mm. Compare with 3531.6 N / 0.5 mm, about 7 kN/mm, for displacement.
- [ ] The frequency constraint is the only one with a nonzero multiplier. What the multiplier means in grams per 1 % tighter limit.
- Where: KKT tables in report section 6.

## 10. Why buckling is inactive
- [ ] The frequency-sized legs are stocky, so the buckling factor is around 230 against a requirement of 3.
- [ ] In the side study without the frequency requirement, buckling becomes active only above about 12.
- [ ] Why the side-study masses are not monotone in the requirement (local optima).
- Where: report section 7.

## 11. Limitations of relaxed stress
- [ ] sigma_relaxed = rho^q sigma_solid with q = 1/2. It equals the true stress only in fully solid voxels.
- [ ] The p-norm is an upper bound on the maximum of the relaxed stress, not of the "real" stress in gray material.
- [ ] Stress at the clamped pad edges and patch corners is singular, so the peak grows with refinement.
- Where: report sections 3, 4 and 10.

## 12. What the 5 mm check proves and does not prove
- [ ] Proves: the same geometry, analysed with voxels half the size, still meets 100 Hz (with margin), stress and displacement.
- [ ] Does not prove: mesh convergence, buckling at 5 mm (not recomputed), or anything about real bolts, contact, fatigue or manufacturing.
- [ ] Why the 6.67 mm check gives less than 100 Hz with both transfer methods, and what that says about our 1 Hz margin on the 5 mm mesh.
- [ ] Why the 5 mm transfer is exact (each voxel split into eight) and the 6.67 mm transfer is volume-weighted.
- Where: report section 8, `src/report.py` (`_upsample`, `transfer_overlap`).

## Before submission
- [ ] Fill in names and contributions in the report (`src/site_template.html`, "Team and contributions") and in the table above.
- [ ] Add the sentence "The student team reviewed the sections identified in the contribution statement." to the AI-use statement only after the reviews are done.
- [ ] Run `python run_all.py --report-only` and check the page locally.
