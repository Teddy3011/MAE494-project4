---
title: Engine-mount bracket design optimization
---

# Engine-mount bracket: PDE-constrained design optimization

MAE 494/598 Project 4. **Work in progress**; the final report is due Nov 25, 2026.

**Structure:** aluminium 6061-T6 engine-mount bracket (base plate, web, ribs, top plate),
3D linear elasticity with 8-node hexahedral elements, 24 design variables
(plate/web/rib thicknesses, rib heights, fillet radii, lightening holes).

**Problem:** minimize mass subject to aggregated von Mises stress &le; 184 MPa,
mount-point displacement &le; 0.5 mm, first natural frequency &ge; 100 Hz and
linear buckling load factor &ge; 3 (nonlinear component N1).
