"""Fill src/site_template.html with generated numbers and tables -> docs/index.html.

Every number on the page comes from the data written by run_all.py. Placeholders are @@key@@;
a missing key raises, and any leftover @@ fails the build.
"""
import html
import re

import numpy as np

from .bracket import CONSTRAINTS
from .config import ROOT
from .report import NAMES, constraint_table

LABEL = {"stress_LC1": "stress LC1", "stress_LC2": "stress LC2", "stress_LC3": "stress LC3", "disp_LC1": "displacement LC1",
         "disp_LC2": "displacement LC2", "disp_LC3": "displacement LC3", "freq": "frequency", "buckling_LC2": "buckling LC2",
         "buckling_LC3": "buckling LC3"}
# approximate change of the scaled constraint g when the physical limit is tightened by 1 %
DG_PER_PCT = {"freq": 0.02, "buckling_LC2": 0.01, "buckling_LC3": 0.01, **{f"stress_LC{k}": 0.01 for k in (1, 2, 3)},
              **{f"disp_LC{k}": 0.01 for k in (1, 2, 3)}}


def f(v, d=3):
    """Number formatting: fixed decimals, or 3 significant digits in scientific notation for tiny/huge."""
    v = float(v)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        m, e = f"{v:.2e}".split("e")
        return f"{m}&times;10<sup>{int(e)}</sup>"
    return f"{v:.{d}f}"


def tab(head, rows, num_from=1):
    h = "".join(f"<th>{c}</th>" for c in head)
    body = "".join("<tr>" + "".join(f'<td class="num">{c}</td>' if i >= num_from else f"<td>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><tr>{h}</tr>{body}</table></div>'


def status(ok):
    return '<span class="ok">&#10003; pass</span>' if ok else '<span class="bad">&#10007; FAIL</span>'


def fig(name, cap):
    return f'<figure><img src="assets/figures/{name}" alt="{html.escape(cap)}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


def build(data, out, model):
    cfg, runs, D = data["cfg"], data["runs"], out["designs"]
    A, B, I = D["A"], D["B"], D["initial"]
    lim = cfg["limits"]
    ctx = {}
    ctx["mode_note"] = "" if data["mode"] == "final" else (
        '<p class="note bad">QUICK MODE OUTPUT: principal runs on the coarse mesh with shortened continuation. '
        "Not for submission; run <code>python run_all.py</code>.</p>")
    ctx["h_mm"] = f"{out['h_report'] * 1e3:.0f}"
    ctx["ne"] = str(B["ne"])
    ctx["n_design"] = str(B["n"])
    ctx["mass_I"], ctx["mass_A"], ctx["mass_B"] = f(I["mass"]), f(A["mass"]), f(B["mass"])
    ctx["reduction"] = f"{100 * (1 - B['mass'] / I['mass']):.1f}"
    ctx["f1_B"] = f(B["freq_Hz"][0], 1)
    ctx["alpha_B"] = f(B["alpha_min"], 1)
    ctx["alpha_A"] = f(A["alpha_min"], 1)
    ctx["vm_B"] = f(B["max_vm_MPa"], 1)
    ctx["disp_B"] = f(max(B["disp_mm"]), 3)
    allB = all(r[4] for r in constraint_table(B))
    allA = all(r[4] for r in constraint_table(A))
    ctx["feasible_B"] = "all constraints satisfied" if allB else "NOT all constraints satisfied (see results table)"
    ctx["kpi_feasible"] = status(allB)
    ctx["dm_AB"] = f"{100 * (B['mass'] - A['mass']) / A['mass']:+.2f}"
    actB = [c for c in CONSTRAINTS if B["g"][c] > -1e-2]
    ctx["active_B"] = ", ".join(LABEL[c] for c in actB) or "none"
    ctx["W"] = f"{cfg['loads']['engine_mass'] * cfg['loads']['g'] * cfg['loads']['g_factor']:.1f}"
    ctx["sig_allow"] = f"{cfg['material']['yield'] / lim['safety_factor'] / 1e6:.0f}"
    ctx["lam_req"] = f"{(2 * np.pi * lim['freq_min']) ** 2:.4g}"

    # results table: initial / A / B
    tI, tA, tB = constraint_table(I), constraint_table(A), constraint_table(B)
    rows = [["Mass [kg]", f(I["mass"]), f(A["mass"]), f(B["mass"]), "minimize"]]
    for (n, vi, l, u, oi), (_, va, _, _, oa), (_, vb, _, _, ob) in zip(tI, tA, tB):
        rel = "&ge;" if n.startswith(("First", "Buck")) else "&le;"
        rows.append([f"{n} [{u}]", f"{f(vi, 2)} {status(oi)}", f"{f(va, 2)} {status(oa)}", f"{f(vb, 2)} {status(ob)}", f"{rel} {f(l, 2)}"])
    rows.append(["Frequencies 1-6 [Hz]"] + [", ".join(f"{v:.1f}" for v in D[k]["freq_Hz"]) for k in ("initial", "A", "B")] + [""])
    rows.append(["Solid fraction (&rho; &ge; 0.5)"] + [f(D[k]["solid_fraction"]) for k in ("initial", "A", "B")] + [""])
    rows.append(["Grayness mean 4&rho;(1-&rho;)"] + [f(D[k]["grayness"]) for k in ("initial", "A", "B")] + [""])
    rows.append(["One analysis, all responses [s]"] + [f(D[k]["seconds"], 2) for k in ("initial", "A", "B")] + [""])
    rows.append(["Optimization wall time [s]", "-", f(runs["A"]["seconds"], 0), f(runs["B"]["seconds"], 0), ""])
    rows.append(["MMA iterations", "-", str(runs["A"]["iterations"]), str(runs["B"]["iterations"]), ""])
    ctx["results_table"] = tab(["Quantity", NAMES["initial"], NAMES["A"], NAMES["B"], "Limit"], rows)

    # verification
    vr = [[r["id"], html.escape(r["name"]), f(r["value"], 6), f(r["exact"], 6), f(r["error"], 6), f(r["tol"], 6), status(r["passed"]),
           html.escape(r["note"])] for r in data["rows"]]
    ctx["verification_table"] = tab(["ID", "Test", "Computed", "Exact / reference", "Error", "Tolerance", "Status", "Note"], vr, 2)
    ctx["n_ver"] = str(len(data["rows"]))
    ctx["n_ver_pass"] = str(sum(r["passed"] for r in data["rows"]))
    mc = data["mesh_conv"]
    ctx["mesh_table"] = tab(["h [mm]", "elements", "DOFs", "mount disp. LC1 [mm]", "f1 [Hz]", "&alpha;<sub>cr</sub> LC2", "max &sigma;<sub>vm</sub> [MPa]", "time [s]"],
                            [[f"{r['h'] * 1e3:.2f}", str(r["ne"]), str(r["ndof"]), f(r["disp_LC1"] * 1e3, 5), f(r["f1"], 2), f(r["alpha_LC2"], 1),
                              f(r["max_vm"] / 1e6, 2), f(r["seconds"], 1)] for r in mc])
    ctx["conv_f1"] = f"{100 * abs(mc[-2]['f1'] / mc[-1]['f1'] - 1):.2f}"
    ctx["conv_disp"] = f"{100 * abs(mc[-2]['disp_LC1'] / mc[-1]['disp_LC1'] - 1):.2f}"
    ctx["conv_vm"] = f"{100 * (mc[-1]['max_vm'] / mc[-2]['max_vm'] - 1):.1f}"

    # gradients
    g = data["grows"]
    by_fn = {}
    for r in g:
        by_fn.setdefault(r["function"], []).append(r)
    grow = []
    for fn, rs in by_fn.items():
        rnd = [r for r in rs if r["direction"] == "random unit"][0]
        grow.append([fn, f(rnd["adjoint"], 8), f(rnd["central_difference"], 8), f(rnd["rel_error"]), f(max(r["rel_error"] for r in rs)),
                     status(all(r["passed"] for r in rs))])
    ctx["grad_table"] = tab(["Function", "Adjoint &nabla;f&middot;d (random d)", "Central difference", "Rel. error (random d)",
                             "Max rel. error, all 4 directions", "Status (tol 10<sup>-5</sup>)"], grow)
    errs = np.array([r["rel_error"] for r in g])
    ctx["grad_max"], ctx["grad_median"] = f(errs.max()), f(np.median(errs))
    ctx["grad_n"] = str(len(g))
    mi = data["method_i"]
    ctx["method_i_table"] = tab(["Function", "Value, sparse (ii)", "Value, dense JAX (i)", "||g<sub>ii</sub>-g<sub>i</sub>||/||g<sub>i</sub>||", "cosine", "Status"],
                                [[r["function"], f(r["value_sparse"], 6), f(r["value_dense"], 6), f(r["rel_error"]), f"{r['cosine']:.12f}", status(r["passed"])] for r in mi])
    ctx["method_i_n"] = str(mi[0]["n"])
    ctx["cost_table"] = tab(["h [mm]", "design variables n", "DOFs", "analysis [s]", "analysis + all 10 gradients [s]", "adjoint overhead [s]",
                             "central FD, 2n analyses [s] (estimate)", "speed-up"],
                            [[f"{c['h'] * 1e3:.0f}", str(c["n"]), str(c["ndof"]), f(c["t_analysis"], 2), f(c["t_analysis_plus_adjoint"], 2),
                              f(c["adjoint_overhead"], 2), f(c["t_fd_central_estimate"], 0), f"{c['speedup']:.0f}&times;"] for c in data["gcost"]])

    # optimization: multistart, KKT
    ms = []
    for k in ["S1", "S2", "S3", "A_dev"]:
        r = runs[k]
        m = r["metrics"]
        ms.append([k, f"{r['h'] * 1e3:.0f}", f"{r['x0']:g}", "yes" if r["use_buckling"] else "no", f(m["mass"]), f(m["freq_Hz"][0], 1),
                   f(m["alpha_min"], 1), f(max(m["g"][CONSTRAINTS[i]] for i in r["active"]), 4), str(r["iterations"]), f(r["seconds"], 0), r["chosen"]])
    ctx["multistart_table"] = tab(["Run", "h [mm]", "x<sub>0</sub>", "buckling", "mass [kg]", "f1 [Hz]", "&alpha;<sub>cr</sub>", "max g", "iterations", "time [s]", "reported iterate"], ms)
    sm = [runs[k]["metrics"]["mass"] for k in ("S1", "S2", "S3")]
    ctx["ms_spread"] = f"{100 * (max(sm) - min(sm)) / min(sm):.1f}"
    kk = []
    for k in ("A", "B"):
        r = runs[k]
        K = r["kkt"]
        for c, lam in K["multipliers"].items():
            dm = lam * DG_PER_PCT[c] * out["mass_solid"] * 1e3
            kk.append([k, LABEL[c], f(D[k]["g"][c], 5), f(lam, 5), f(r["mma_multipliers"][c], 5), f(dm, 1)])
    ctx["kkt_mult_table"] = tab(["Run", "Near-active constraint", "g at optimum", "&lambda; (NNLS stationarity)", "&lambda; (MMA dual)",
                                 "approx. mass cost of a 1 % tighter limit [g]"], kk)
    ctx["kkt_res_table"] = tab(["Run", "stationarity ||&nabla;f + J<sup>T</sup>&lambda;||/||&nabla;f|| (free vars)", "bound-sign violation",
                                "primal violation max(g,0)", "complementarity max|&lambda; g|", "vars at lower bound", "at upper bound", "between bounds"],
                               [[k, f(runs[k]["kkt"]["stationarity"], 4), f(runs[k]["kkt"]["bound_sign_violation"], 4), f(runs[k]["kkt"]["primal_violation"], 5),
                                 f(runs[k]["kkt"]["complementarity"], 6), str(runs[k]["kkt"]["n_lower"]), str(runs[k]["kkt"]["n_upper"]), str(runs[k]["kkt"]["n_inner"])]
                                for k in ("A", "B")])
    ctx["stat_B"] = f(runs["B"]["kkt"]["stationarity"], 3)
    ctx["chosen_A"], ctx["chosen_B"] = runs["A"]["chosen"], runs["B"]["chosen"]
    ctx["cont_table"] = tab(["Stage", "SIMP p", "projection &beta;", "max MMA iterations", "move limit"],
                            [[str(i + 1), f"{p:g}", f"{b:g}", str(int(n)), f"{cfg['optimizer']['move_final'] if i == len(cfg['simp']['continuation']) - 1 else cfg['optimizer']['move']:g}"]
                             for i, (p, b, n) in enumerate(cfg["simp"]["continuation"])])

    # nonlinear study
    nl = []
    for k in ["N", "NB"] + [k for k in runs if k.startswith("NB") and k != "NB"]:
        r = runs[k]
        m = r["metrics"]
        nl.append([k, "none" if not r["use_buckling"] else f"&ge; {r['alpha_req']:g}", f(m["mass"]), f(m["alpha_min"], 2),
                   f(max(m["pnorm_MPa"]), 1), f(max(m["disp_mm"]), 3), f(m["freq_Hz"][0], 1),
                   ", ".join(LABEL[c] for c in CONSTRAINTS if CONSTRAINTS.index(c) in r["active"] and m["g"][c] > -1e-2) or "none"])
    ctx["nl_table"] = tab(["Run", "buckling requirement", "mass [kg]", "&alpha;<sub>cr</sub>", "max stress p-norm [MPa]", "max mount disp. [mm]", "f1 [Hz]", "near-active constraints"], nl)
    ctx["alpha_N"] = f(runs["N"]["metrics"]["alpha_min"], 1)
    ctx["mass_N"] = f(runs["N"]["metrics"]["mass"])

    # checks
    ch = out["checks"]
    crow = []
    for key, lab in [("refined_S1", "20 mm design (S1) upsampled to the 10 mm mesh"), ("thresholded_B", "Design B thresholded at &rho; = 0.5 (0/1 design)")]:
        if key in ch:
            m = ch[key]
            crow.append([lab, f(m["mass"]), f(m["freq_Hz"][0], 1), f(m["alpha_min"], 1), f(max(m["pnorm_MPa"]), 1), f(m["max_vm_MPa"], 1), f(max(m["disp_mm"]), 3)])
    crow.append(["Reference: design B as analysed", f(B["mass"]), f(B["freq_Hz"][0], 1), f(B["alpha_min"], 1), f(max(B["pnorm_MPa"]), 1), f(B["max_vm_MPa"], 1), f(max(B["disp_mm"]), 3)])
    ctx["checks_table"] = tab(["Check", "mass [kg]", "f1 [Hz]", "&alpha;<sub>cr</sub>", "max p-norm [MPa]", "max &sigma;<sub>vm</sub> [MPa]", "max mount disp. [mm]"], crow)
    if "refined_S1" in ch:
        ctx["refine_f1"] = f(ch["refined_S1"]["freq_Hz"][0], 1)
        ctx["coarse_f1"] = f(runs["S1"]["metrics"]["freq_Hz"][0], 1)
    else:
        ctx["refine_f1"] = ctx["coarse_f1"] = "n/a (quick mode)"
    ctx["thr_f1"] = f(ch["thresholded_B"]["freq_Hz"][0], 1)

    figs = dict(
        fig_verification=("verification_convergence.png", "V3 and V4 relative error vs element length; bracket responses on the solid domain vs voxel size, normalized by the finest mesh."),
        fig_taylor=("gradient_taylor.png", "Random-direction gradient check for all ten functions: central-difference error and Taylor remainder vs step; dashed line slope 2."),
        fig_history=("optimization_history.png", "Mass history of the principal runs (A, B) and the multistart runs. Vertical lines: continuation steps."),
        fig_constraints=("constraint_history.png", "Scaled constraints of run B during the optimization (feasible below zero)."),
        fig_buckling=("buckling_comparison.png", "Left: critical buckling factor of the reported designs (log scale). Right: supplementary study without the frequency requirement."),
        fig_geometry=("initial_vs_optimized.png", "Initial and optimized designs (voxels with density at least 0.5)."),
        fig_stress=("stress_comparison.png", "Stress utilization, maximum over the three load cases (cividis scale)."),
        fig_modes=("modes.png", "First six natural frequencies with the engine as lumped mass; dashed line: 100 Hz requirement."))
    for k, (name, cap) in figs.items():
        ctx[k] = fig(name, cap)
    ctx["t_pipeline"] = f"{data['t_pipeline'] / 60:.0f}"
    import json
    env = json.loads((ROOT / "results" / "environment.json").read_text())
    ctx["env"] = html.escape(f"{env['platform']}, {env['processor']}, {env['cpu_count']} logical CPUs, Python {env['python'].split()[0]}")

    tpl = (ROOT / "src" / "site_template.html").read_text(encoding="utf-8")
    page = re.sub(r"@@(\w+)@@", lambda m: ctx[m.group(1)], tpl)
    if "@@" in page:
        raise ValueError("unfilled placeholder in site template")
    (ROOT / "docs" / "index.html").write_text(page, encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("")
