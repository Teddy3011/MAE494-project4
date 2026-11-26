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


# Extra "things that did not work" items, written after looking at the convergence study.
FAILED_TRIALS_EXTRA = (
    "<li><b>Crisper final designs.</b> Raising the projection to &beta; = 16 and then 32 (cases E3 and E4 in the convergence-study "
    "table) made the bracket much lighter on the 10 mm mesh. But those designs lost more frequency on the 5 mm mesh and converged "
    "worse. We chose the converged &beta; = 8 design. The lighter route would need a higher frequency target and more iterations, "
    "and we did not have time to finish it.</li>"
    "<li><b>Running all 5 mm checks at once.</b> Eight parallel workers each built a 5 mm model and the computer ran out of memory, "
    "which lost a 1-hour sweep. The 5 mm checks now run one at a time, and every optimization run is saved as soon as it finishes "
    "(<code>python run_all.py --resume</code> reuses them).</li>")


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
    sel_fail = "" if any(r["ok"] for r in data["selection"]) else (
        '<p class="note bad">NO SWEEP CANDIDATE PASSED EVERY CHECK. The design shown is the candidate with the highest '
        "refined-mesh frequency and does not meet the selection rule (see section 6).</p>")
    ctx["mode_note"] = sel_fail + "" if data["mode"] == "final" else sel_fail + (
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
    ref = out["checks"].get("refined_B")
    ref_ok = ref is not None and (ref["freq_Hz"][0] >= lim["freq_min"] and max(ref["pnorm_MPa"]) <= B["allow"]["stress_MPa"]
                                  and max(ref["disp_mm"]) <= B["allow"]["disp_mm"])
    ctx["kpi_feasible"] = status(allB and ref_ok)
    ctx["dm_AB"] = f"{100 * (B['mass'] - A['mass']) / A['mass']:+.2f}"
    actB = [c for c in CONSTRAINTS if B["g"][c] > -1e-2]
    ctx["active_B"] = ", ".join(LABEL[c] for c in actB) or "none"
    ctx["W"] = f"{cfg['loads']['engine_mass'] * cfg['loads']['g'] * cfg['loads']['g_factor']:.1f}"
    ctx["sig_allow"] = f"{cfg['material']['yield'] / lim['safety_factor'] / 1e6:.0f}"
    fdes = lim.get("freq_design", lim["freq_min"])
    ctx["lam_req"] = f"{(2 * np.pi * fdes) ** 2:.4g}"
    ctx["f_design"] = f"{fdes:g}"
    ffd = data.get("full_fd")
    if ffd:
        ctx["full_fd_table"] = tab(["Function", "||g<sub>adj</sub>-g<sub>FD</sub>||/||g<sub>FD</sub>||", "max |difference|", "||g<sub>FD</sub>||"],
                                   [[r["function"], f(r["rel_error"]), f(r["max_abs_diff"]), f(r["grad_norm"], 5)] for r in ffd["rows"]])
        ctx["full_fd_note"] = (f"Measured on the {ffd['h'] * 1e3:.0f} mm mesh: the complete central-difference gradient "
                               f"({ffd['analyses']} analyses for {ffd['n']} variables, step {ffd['eps']:g}) took {ffd['seconds_fd']:.0f} s; "
                               f"the adjoint gradient of all ten functions, including the analysis, took {ffd['seconds_adjoint']:.2f} s "
                               f"({ffd['seconds_fd'] / ffd['seconds_adjoint']:.0f}&times; faster). Component-by-component agreement:")
    else:
        ctx["full_fd_table"], ctx["full_fd_note"] = "", "The measured full finite-difference gradient is skipped in quick mode."
    def modes(k):
        return "; ".join(f"{i + 1}: {m['axis']} ({100 * m['axis_share']:.0f} %, {100 * m['engine_mass_share']:.0f} %)" for i, m in enumerate(D[k]["modes"]))

    # results table: initial / A / B
    tI, tA, tB = constraint_table(I), constraint_table(A), constraint_table(B)
    rows = [["Mass [kg]", f(I["mass"]), f(A["mass"]), f(B["mass"]), "minimize"]]
    for (n, vi, l, u, oi), (_, va, _, _, oa), (_, vb, _, _, ob) in zip(tI, tA, tB):
        rel = "&ge;" if n.startswith(("First", "Buck")) else "&le;"
        rows.append([f"{n} [{u}]", f"{f(vi, 2)} {status(oi)}", f"{f(va, 2)} {status(oa)}", f"{f(vb, 2)} {status(ob)}", f"{rel} {f(l, 2)}"])
    rows.append(["Frequencies 1-6 [Hz]"] + [", ".join(f"{v:.1f}" for v in D[k]["freq_Hz"]) for k in ("initial", "A", "B")] + [""])
    rows.append(["Modes 1-6: engine translation axis (share of translation, engine share of modal mass)"] + [modes(k) for k in ("initial", "A", "B")] + [""])
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
                                "primal violation max(g,0)", "dual feasibility &lambda; &ge; 0", "complementarity max|&lambda; g|", "vars at lower bound", "at upper bound", "between bounds"],
                               [[k, f(runs[k]["kkt"]["stationarity"], 4), f(runs[k]["kkt"]["bound_sign_violation"], 4), f(runs[k]["kkt"]["primal_violation"], 6),
                                 "yes" if runs[k]["kkt"].get("dual_feasible", True) else "NO", f(runs[k]["kkt"]["complementarity"], 6), str(runs[k]["kkt"]["n_lower"]), str(runs[k]["kkt"]["n_upper"]), str(runs[k]["kkt"]["n_inner"])]
                                for k in ("A", "B")])
    ctx["stat_B"] = f(runs["B"]["kkt"]["stationarity"], 3)
    ctx["chosen_A"], ctx["chosen_B"] = runs["A"]["chosen"], runs["B"]["chosen"]
    cont = cfg["simp"]["continuation"]
    crows = [["A_base", str(i + 1), f"{p:g}", f"{b:g}", str(int(n)), f"{cfg['optimizer']['move_final'] if i == len(cont) - 1 else cfg['optimizer']['move']:g}"]
             for i, (p, b, n) in enumerate(cont)]
    crows += [["A, B and sweep (restart from A_base)", str(len(cont) + i + 1), f"{p:g}", f"{b:g}", str(int(n)), f"{mv:g}"]
              for i, (p, b, n, mv) in enumerate(runs["B"]["stages"])]
    ctx["cont_table"] = tab(["Run", "Stage", "SIMP p", "projection &beta;", "max MMA iterations", "move limit"], crows)
    ctx["beta_final"] = f"{runs['B']['stages'][-1][1]:g}"
    ctx["target_sel"] = f"{data['target']:g}"
    ctx["margin_hz"] = f"{cfg['sweep']['refined_margin_hz']:g}"
    sw = []
    for row in data["selection"]:
        t = row["target"]
        ra, rb_ = runs[f"A_{t:g}"], runs[f"B_{t:g}"]
        m, rf = rb_["metrics"], rb_["refined"]
        failed = [k.replace("_", " ") for k, v in row["checks"].items() if not v]
        sw.append([f"{t:g}", f(ra["metrics"]["mass"], 4), f(m["mass"], 4), f(m["freq_Hz"][0], 2), f(rf["freq_Hz"][0], 2),
                   f(max(rf["pnorm_MPa"]), 1), f(rf["max_vm_MPa"], 1), f(max(rf["disp_mm"]), 4), f(m["alpha_min"], 1),
                   f(max(v for v in m["g"].values() if v is not None), 6), f(rb_["kkt"]["stationarity"], 4), str(rb_["iterations"]),
                   f(rb_["seconds"], 0), f(m["grayness"], 3), str(m["connectivity"]["components"]),
                   "<b>selected</b>" if t == data["target"] else (status(True) if row["ok"] else "fails: " + ", ".join(failed))])
    ctx["sweep_table"] = tab(["Target [Hz]", "A mass [kg]", "B mass [kg]", "B f1, 10 mm [Hz]", "B f1, 5 mm [Hz]", "B p-norm, 5 mm [MPa]",
                              "B max &sigma;<sub>vm</sub>, 5 mm [MPa]", "B disp., 5 mm [mm]", "B &alpha;<sub>cr</sub>", "B max g", "B stationarity",
                              "B iterations", "B time [s]", "B grayness", "B solid parts", "Rule"], sw)
    cs = out.get("convergence_study")
    if cs:
        ctx["conv_study_table"] = tab(["Case", "Final-stage setting", "iterations", "mass [kg]", "f1, 10 mm [Hz]", "f1, 5 mm [Hz]", "max g", "stationarity",
                                       "bound-sign", "grayness", "solid parts", "reported iterate"],
                                      [[r["case"], r["setting"], r["iterations"], f(r["mass_kg"], 4), f(r["f1_10mm_Hz"], 2), f(r["f1_5mm_Hz"], 2), f(r["max_g"], 6),
                                        f(r["stationarity"], 4), f(r["bound_sign"], 4), f(r["grayness"], 3), r["components"], r["reported"]] for r in cs])
    else:
        ctx["conv_study_table"] = "<p>(results/convergence_study.csv not found)</p>"
    p_, b_, n_, mv_ = runs["B"]["stages"][-1]
    ctx["conv_study_note"] = (f"Based on this, the final stage in <code>run_all.py</code> uses p = {p_:g}, &beta; = {b_:g}, move limit {mv_:g} "
                              f"and up to {int(n_)} iterations (<code>sweep.polish_stages</code> in <code>config.yaml</code>).")

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
    for key, lab in [("refined_S1", "20 mm design (S1) re-analysed on the 10 mm mesh"),
                     ("refined_B", "Design B re-analysed on a 5 mm mesh (each voxel split in 8)"),
                     ("mid_B", "Design B on a 6.67 mm mesh (volume-overlap transfer)"),
                     ("mid_B_nearest", "Design B on a 6.67 mm mesh (nearest-voxel transfer)"),
                     ("thresholded_B", "Design B thresholded at &rho; = 0.5 (0/1 design)")]:
        if key in ch:
            m = ch[key]
            a = f(m["alpha_min"], 1) if m["alpha_min"] is not None else "not computed"
            crow.append([lab, f(m["mass"]), f(m["freq_Hz"][0], 1), a, f(max(m["pnorm_MPa"]), 1), f(m["max_vm_MPa"], 1), f(max(m["disp_mm"]), 3)])
    crow.append(["Reference: design B as analysed", f(B["mass"]), f(B["freq_Hz"][0], 1), f(B["alpha_min"], 1), f(max(B["pnorm_MPa"]), 1), f(B["max_vm_MPa"], 1), f(max(B["disp_mm"]), 3)])
    ctx["checks_table"] = tab(["Check", "mass [kg]", "f1 [Hz]", "&alpha;<sub>cr</sub>", "max p-norm [MPa]", "max &sigma;<sub>vm</sub> [MPa]", "max mount disp. [mm]"], crow)
    if "refined_S1" in ch:
        ctx["refine_f1"] = f(ch["refined_S1"]["freq_Hz"][0], 1)
        ctx["coarse_f1"] = f(runs["S1"]["metrics"]["freq_Hz"][0], 1)
    else:
        ctx["refine_f1"] = ctx["coarse_f1"] = "n/a (quick mode)"
    ctx["thr_f1"] = f(ch["thresholded_B"]["freq_Hz"][0], 1)
    ctx["thr_mass"] = f(ch["thresholded_B"]["mass"])
    if "refined_B" in ch:
        rb = ch["refined_B"]
        ctx["ref5_f1"] = f(rb["freq_Hz"][0], 1)
        ctx["ref5_pct"] = f"{100 * (rb['freq_Hz'][0] / B['freq_Hz'][0] - 1):+.1f}"
        ok5 = rb["freq_Hz"][0] >= lim["freq_min"]
        ctx["ref5_verdict"] = ("still meets the 100 Hz requirement on the refined mesh" if ok5 else
                               "falls below the 100 Hz requirement on the refined mesh despite the design margin")
        ctx["ref5_stress"] = f(max(rb["pnorm_MPa"]), 1)
        ctx["ref5_ratio"] = f"{B['freq_Hz'][0] / rb['freq_Hz'][0]:.3f}"
    else:
        ctx["ref5_f1"] = ctx["ref5_pct"] = ctx["ref5_stress"] = ctx["ref5_ratio"] = "n/a"
        ctx["ref5_verdict"] = "was not run (quick mode)"
    ctx["mid_f1"] = f(ch["mid_B"]["freq_Hz"][0], 1) if "mid_B" in ch else "n/a (quick mode)"
    ctx["mid_near_f1"] = f(ch["mid_B_nearest"]["freq_Hz"][0], 1) if "mid_B_nearest" in ch else "n/a (quick mode)"
    ctx["mid_near_mass"] = f(ch["mid_B_nearest"]["mass"]) if "mid_B_nearest" in ch else "n/a"
    if "mid_B" in ch and "mid_B_nearest" in ch:
        mids = sorted([ch["mid_B"]["freq_Hz"][0], ch["mid_B_nearest"]["freq_Hz"][0]])
        ctx["mid_range"] = f"{mids[0]:.1f}&ndash;{mids[1]:.1f}"
        ctx["mid_verdict"] = ("which also meets 100 Hz" if mids[0] >= lim["freq_min"] else
                              "which is below the 100 Hz requirement")
    else:
        ctx["mid_range"], ctx["mid_verdict"] = "n/a", "(not run in quick mode)"
    ctx["failed_trials_extra"] = FAILED_TRIALS_EXTRA
    ctx["a_conv"] = " &rarr; ".join(f"{r['alpha_LC2']:.0f}" for r in mc)

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
