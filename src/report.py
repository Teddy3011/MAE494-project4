"""Post-processing: independent re-analysis of the reported designs, refinement and threshold
checks, metrics.json, results.json (report + 3D viewer), optimization history and figures."""
import csv
import json
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .bracket import Bracket, CONSTRAINTS  # noqa: E402
from .config import ROOT  # noqa: E402

RES, FIG, DOCS = ROOT / "results", ROOT / "results" / "figures", ROOT / "docs"
NAMES = {"initial": "Initial (solid domain)", "A": "Optimized, no buckling constraint", "B": "Optimized, with buckling constraint"}
INK, MUTED = "#1f2328", "#6e7781"
C_A, C_B, C_I = "#0b6e99", "#c2410c", "#6e7781"


def metrics(model, r):
    """Physical values of one analysis, with every constraint and its pass/fail status."""
    lim = model.cfg["limits"]
    disp = np.linalg.norm(r["disp"], axis=1)
    alpha = {f"LC{k + 1}": (1.0 / r["mu"][k]).tolist() for k in r["mu"]}
    g = dict(zip(CONSTRAINTS, map(float, r["g"])))
    return dict(
        mass=float(r["mass"]), mass_fraction=float(r["f"]),
        g=g, pnorm_MPa=(r["pnorm"] / 1e6).tolist(), max_vm_MPa=float(r["vm_max"].max() / 1e6),
        max_vm_by_lc_MPa=(r["vm_max"].max(0) / 1e6).tolist(), disp_mm=(disp * 1e3).tolist(),
        freq_Hz=r["freq"].tolist(), alpha=alpha,
        alpha_min=float(min(a[0] for a in alpha.values())) if alpha else None,
        allow=dict(stress_MPa=model.sig_allow / 1e6, disp_mm=lim["disp_max"] * 1e3, freq_Hz=lim["freq_min"],
                   alpha=lim["buckling_min"]),
        grayness=float(np.mean(4 * r["rho"] * (1 - r["rho"]))),
        solid_fraction=float(np.mean(r["rho"] >= 0.5)),
        seconds=float(sum(r["timings"].values())), timings={k: float(v) for k, v in r["timings"].items()},
        ne=int(model.ne), n=int(model.n), h=float(model.h))


def constraint_table(m, alpha_req=None):
    """Rows (name, value, limit, unit, passed) on true values, not on the smooth aggregates."""
    a = m["allow"]
    areq = alpha_req or a["alpha"]
    rows = []
    for k in range(3):
        rows.append((f"Stress p-norm LC{k + 1}", m["pnorm_MPa"][k], a["stress_MPa"], "MPa", m["g"][f"stress_LC{k + 1}"] <= 1e-3))
    rows.append(("True max von Mises (relaxed)", m["max_vm_MPa"], a["stress_MPa"], "MPa", m["max_vm_MPa"] <= a["stress_MPa"]))
    for k in range(3):
        rows.append((f"Mount displacement LC{k + 1}", m["disp_mm"][k], a["disp_mm"], "mm", m["disp_mm"][k] <= a["disp_mm"] * (1 + 1e-3)))
    rows.append(("First natural frequency", m["freq_Hz"][0], a["freq_Hz"], "Hz", m["freq_Hz"][0] >= a["freq_Hz"] * (1 - 5e-4)))
    for lc in ("LC2", "LC3"):
        rows.append((f"Buckling factor {lc}", m["alpha"][lc][0], areq, "-", m["alpha"][lc][0] >= areq * (1 - 1e-3)))
    return rows


def _upsample(rho, nel_c):
    nx, ny, nz = nel_c
    r = rho.reshape(nz, ny, nx)
    return np.repeat(np.repeat(np.repeat(r, 2, 0), 2, 1), 2, 2).ravel()


def _norm_field(v, digits=4):
    s = float(np.max(np.abs(v))) or 1.0
    return dict(scale=s, data=np.round(v / s, digits).tolist())


def build(data):
    cfg, runs = data["cfg"], data["runs"]
    FIG.mkdir(parents=True, exist_ok=True)
    (DOCS / "assets" / "figures").mkdir(parents=True, exist_ok=True)
    p, beta = cfg["simp"]["continuation"][-1][:2]
    h = runs["B"]["h"]
    model = Bracket(cfg, h)

    # independent re-analysis of the three reported designs on the reporting mesh
    designs = {"initial": np.ones(model.n), "A": runs["A"]["x"], "B": runs["B"]["x"]}
    res, met = {}, {}
    for k, x in designs.items():
        res[k] = model.analyze(x, p, beta, grad=False)
        met[k] = metrics(model, res[k])

    # mesh check: 20 mm with-buckling design (start S1) upsampled 2x and re-analysed at 10 mm
    checks = {}
    if runs["S1"]["h"] == 2 * h:
        coarse = Bracket(cfg, runs["S1"]["h"])
        r = model.analyze(None, p, beta, grad=False, rho=_upsample(runs["S1"]["rho"], coarse.nel))
        checks["refined_S1"] = metrics(model, r)
    # manufacturing interpretation: threshold B at rho = 0.5 and re-analyse the 0/1 design
    rb = np.where(res["B"]["rho"] >= 0.5, 1.0, cfg["simp"]["x_min"])
    checks["thresholded_B"] = metrics(model, model.analyze(None, p, beta, grad=False, rho=rb))

    # tables
    hist = [dict(run=k, **row) for k, rr in runs.items() for row in rr["history"]]
    _csv(hist, RES / "optimization_history.csv")
    summary = {k: dict(h=rr["h"], x0=rr["x0"], use_buckling=rr["use_buckling"], use_freq=rr["use_freq"],
                       alpha_req=rr["alpha_req"], iterations=rr["iterations"], seconds=rr["seconds"], chosen=rr["chosen"],
                       kkt=rr["kkt"], mma_multipliers=rr["mma_multipliers"], metrics=rr["metrics"])
               for k, rr in runs.items()}
    out = dict(mode=data["mode"], h_report=h, designs=met, checks=checks, runs=summary,
               gradient_cost=data["gcost"], mesh_convergence=data["mesh_conv"], pipeline_seconds=data["t_pipeline"],
               mass_solid=float(model.mass_solid))
    (RES / "metrics.json").write_text(json.dumps(out, indent=1))

    viewer = viewer_json(model, res, met, runs)
    txt = json.dumps(viewer, separators=(",", ":"))
    (RES / "results.json").write_text(txt)
    (DOCS / "assets" / "results.json").write_text(txt)

    figures(data, model, res, met)
    for f in FIG.glob("*.png"):
        shutil.copy(f, DOCS / "assets" / "figures" / f.name)
    from . import site
    site.build(data, out, model)
    return out


def _csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def viewer_json(model, res, met, runs):
    """Everything the 3D viewer needs, from the same analyses as the report tables."""
    sa = model.sig_allow
    designs = {}
    for k, r in res.items():
        util = r["vm_max"].max(1) / sa
        psi = model.full(r["psi"][1][:, 0])
        designs[k] = dict(
            name=NAMES[k], mass=met[k]["mass"],
            rho=np.round(r["rho"], 3).tolist(), util=np.round(util, 4).tolist(),
            u={f"LC{i + 1}": _norm_field(r["U"][:, i]) for i in range(3)},
            mode1=_norm_field(model.full(r["phi"][:, 0])), buck1=_norm_field(psi),
            mode1_Hz=met[k]["freq_Hz"][0], buck1_alpha=met[k]["alpha"]["LC2"][0],
            table=[dict(name=n, value=v, limit=l, unit=u, passed=bool(ok)) for n, v, l, u, ok in constraint_table(met[k])])
    return dict(grid=dict(nel=list(model.nel), h=model.h, L=model.L.tolist()),
                supports=model.support_nodes.tolist(), interface=model.iface_nodes.tolist(),
                threshold=0.5, limits=met["initial"]["allow"], designs=designs,
                notes="Densities are the analysed (filtered, projected) field; the display threshold only hides voxels.")


# ------------------------------------------------------------------ figures

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)


def figures(data, model, res, met):
    runs = data["runs"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK})

    # 1. verification convergence
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    E, rho = data["cfg"]["material"]["E"], data["cfg"]["material"]["rho"]
    f_ex = 1.8751040687119611 ** 2 / (2 * np.pi * 0.2 ** 2) * np.sqrt(E * 1e-8 / 12 / (rho * 1e-4))
    P_ex = np.pi ** 2 * E * 1e-8 / 12 / 0.8 ** 2
    for a, key, ex, title in [(ax[0], "modal", f_ex, "V3 cantilever $f_1$"), (ax[1], "buckling", P_ex, "V4 Euler column $P_{cr}$")]:
        hh, v = np.array(data["conv"][key]).T
        a.loglog(hh * 1e3, np.abs(v - ex) / ex, "o-", color=C_A)
        a.set_xlabel("element length along the beam [mm]")
        a.set_ylabel("relative error vs theory")
        a.set_title(title)
        _style(a)
    mc = data["mesh_conv"]
    hh = np.array([r["h"] for r in mc]) * 1e3
    for key, lab, c in [("disp_LC1", "mount displacement LC1", C_A), ("f1", "first frequency", C_B),
                        ("alpha_LC2", "buckling factor LC2", "#15803d"), ("max_vm", "max von Mises (singular)", C_I)]:
        v = np.array([r[key] for r in mc])
        ax[2].plot(hh, v / v[-1], "o-", label=lab, color=c)
    ax[2].set_xlabel("voxel size h [mm]")
    ax[2].set_ylabel("value / value at finest h")
    ax[2].set_title("Bracket, solid domain")
    ax[2].invert_xaxis()
    ax[2].legend(fontsize=7, frameon=False)
    _style(ax[2])
    fig.tight_layout()
    fig.savefig(FIG / "verification_convergence.png", dpi=150)
    plt.close(fig)

    # 2. gradient Taylor / step sweep
    sw = [s for s in data["sweep"] if s["direction"] == "random unit"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    cmap = plt.get_cmap("tab10")
    for i, fn in enumerate(["mass"] + CONSTRAINTS):
        s = [r for r in sw if r["function"] == fn]
        e = np.array([r["eps"] for r in s])
        ax[0].loglog(e, np.maximum([r["rel_error"] for r in s], 1e-17), "-", lw=1, color=cmap(i), label=fn)
        ax[1].loglog(e, np.maximum([r["taylor_remainder"] for r in s], 1e-20), "-", lw=1, color=cmap(i))
    e = np.logspace(-1, -4, 5)
    ax[0].loglog(e, 1e-2 * (e / 1e-1) ** 2, "k--", lw=1, label="slope 2")
    ax[1].loglog(e, 1e-2 * (e / 1e-1) ** 2, "k--", lw=1)
    ax[0].set_title("Central difference vs adjoint: relative error")
    ax[1].set_title(r"Taylor remainder $|f(x+\epsilon d)-f(x)-\epsilon\,\nabla f\cdot d|$")
    for a in ax:
        a.set_xlabel(r"step $\epsilon$")
        _style(a)
    ax[0].legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "gradient_taylor.png", dpi=150)
    plt.close(fig)

    # 3. optimization history (mass)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, c, ls in [("A", C_A, "-"), ("B", C_B, "--"), ("S1", MUTED, ":"), ("S2", "#9ca3af", ":"), ("S3", "#4b5563", ":")]:
        hst = runs[k]["history"]
        ax.plot([r["iter"] for r in hst], [r["mass"] for r in hst], ls, color=c, lw=1.4,
                label=f"{k} (h={runs[k]['h'] * 1e3:.0f} mm, x0={runs[k]['x0']:g})")
    for r0, r1 in zip(runs["B"]["history"][:-1], runs["B"]["history"][1:]):
        if r1["stage"] != r0["stage"]:
            ax.axvline(r1["iter"], color="#d0d7de", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("MMA iteration (vertical lines: continuation steps of B)")
    ax.set_ylabel("mass [kg]")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "optimization_history.png", dpi=150)
    plt.close(fig)

    # 4. constraint history of B
    fig, ax = plt.subplots(figsize=(8, 3.4))
    hst = runs["B"]["history"]
    it = [r["iter"] for r in hst]
    for key, lab, c in [("freq", "frequency", C_B), ("stress_LC2", "stress LC2", C_A), ("disp_LC3", "displacement LC3", "#15803d"),
                        ("buckling_LC2", "buckling LC2", C_I)]:
        ax.plot(it, [r[key] for r in hst], lw=1.2, color=c, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("MMA iteration")
    ax.set_ylabel(r"scaled constraint $g_i$ (feasible $\leq 0$)")
    ax.set_title("Run B: constraint history")
    ax.legend(fontsize=7, frameon=False, ncol=4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "constraint_history.png", dpi=150)
    plt.close(fig)

    # 5/6. voxel renders: geometry and stress utilization
    nx, ny, nz = model.nel
    from matplotlib import cm, colors
    norm = colors.Normalize(0, 1)
    for fname, color_by in [("initial_vs_optimized.png", None), ("stress_comparison.png", "util")]:
        fig = plt.figure(figsize=(12, 4))
        for i, k in enumerate(["initial", "A", "B"]):
            a = fig.add_subplot(1, 3, i + 1, projection="3d")
            rho = res[k]["rho"].reshape(nz, ny, nx).transpose(2, 1, 0)
            filled = rho >= 0.5
            if color_by:
                u = (res[k]["vm_max"].max(1) / model.sig_allow).reshape(nz, ny, nx).transpose(2, 1, 0)
                fc = cm.cividis(norm(u))
            else:
                fc = np.empty(filled.shape + (4,))
                fc[:] = colors.to_rgba(C_I if k == "initial" else (C_A if k == "A" else C_B), 0.9)
            a.voxels(filled, facecolors=fc, edgecolor=(0, 0, 0, 0.08), linewidth=0.2)
            a.set_box_aspect((nx, ny, nz))
            a.set_title(f"{NAMES[k]}\n{met[k]['mass']:.3f} kg", fontsize=9)
            a.set_axis_off()
            a.view_init(22, -60)
        if color_by:
            cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap="cividis"), cax=cax, label=r"$\sigma_{vm}/\sigma_{allow}$, max over LC")
        fig.text(0.01, 0.02, "Voxels with physical density >= 0.5 shown; analysis used the continuous density field.",
                 fontsize=7, color=MUTED)
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    # 7. modes
    fig, ax = plt.subplots(figsize=(7, 3.3))
    w = 0.27
    for i, (k, c) in enumerate([("initial", C_I), ("A", C_A), ("B", C_B)]):
        f = met[k]["freq_Hz"]
        ax.bar(np.arange(len(f)) + (i - 1) * w, f, w, color=c, label=NAMES[k])
    ax.axhline(data["cfg"]["limits"]["freq_min"], color=INK, ls="--", lw=1, label="requirement 100 Hz")
    ax.set_xticks(range(6), [f"mode {i + 1}" for i in range(6)])
    ax.set_ylabel("natural frequency [Hz]")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "modes.png", dpi=150)
    plt.close(fig)

    # 8. buckling comparison and supplementary sweep
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    labs = ["initial", "A", "B"]
    av = [met[k]["alpha_min"] for k in labs]
    ax[0].bar(range(3), av, color=[C_I, C_A, C_B])
    ax[0].set_yscale("log")
    ax[0].axhline(data["cfg"]["limits"]["buckling_min"], color=INK, ls="--", lw=1, label="requirement 3")
    ax[0].set_xticks(range(3), ["initial", "A: no buckling", "B: with buckling"])
    ax[0].set_ylabel(r"critical load factor $\alpha_{cr}$ (min over LC2, LC3)")
    ax[0].set_title("Main problem")
    ax[0].legend(fontsize=7, frameon=False)
    _style(ax[0])
    sw = [runs["N"]] + [runs["NB"]] + [runs[k] for k in runs if k.startswith("NB") and k != "NB"]
    req = [0.0] + [r["alpha_req"] for r in sw[1:]]
    ax[1].plot(req, [r["metrics"]["mass"] for r in sw], "o-", color=C_B, label="mass [kg]")
    ax2 = ax[1].twinx()
    ax2.plot(req, [r["metrics"]["alpha_min"] for r in sw], "s--", color=C_A, label=r"achieved $\alpha_{cr}$")
    ax2.plot([0, max(req)], [0, max(req)], color=MUTED, lw=0.8)
    ax[1].set_xlabel(r"required $\alpha_{cr}$ (0 = no buckling constraint)")
    ax[1].set_ylabel("mass [kg]", color=C_B)
    ax2.set_ylabel(r"achieved $\alpha_{cr}$", color=C_A)
    ax[1].set_title("Supplementary: no frequency requirement (20 mm mesh)")
    _style(ax[1])
    fig.tight_layout()
    fig.savefig(FIG / "buckling_comparison.png", dpi=150)
    plt.close(fig)
