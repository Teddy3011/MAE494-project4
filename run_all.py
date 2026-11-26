"""Regenerate every result, figure and the report site from source.

    python run_all.py           final mode: principal optimizations on the 10 mm mesh (about 1 h)
    python run_all.py --quick   pipeline check: everything on the 20 mm mesh, shortened continuation

All numbers in results/ and docs/ come from this script. Generated files are overwritten.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version

import numpy as np

from src.config import ROOT, load

RES = ROOT / "results"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def environment(mode):
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                cpu_count=os.cpu_count(), mode=mode,
                packages={p: version(p) for p in ["numpy", "scipy", "jax", "jaxlib", "PyYAML", "matplotlib", "pytest"]})


def phase1(cfg, quick):
    """Base design on the reporting mesh (full continuation, no buckling) and the 20 mm studies."""
    hd = cfg["mesh"]["h_dev"]
    hr = hd if quick else cfg["mesh"]["h_report"]
    J = {"A_base": dict(h=hr, x0=1.0, buck=False)}
    for i, x0 in enumerate(cfg["optimizer"]["starts"]):
        J[f"S{i + 1}"] = dict(h=hd, x0=x0, buck=True)
    J["A_dev"] = dict(h=hd, x0=1.0, buck=False)
    J["N"] = dict(h=hd, x0=1.0, buck=False, freq=False)
    J["NB"] = dict(h=hd, x0=1.0, buck=True, freq=False)
    for a in cfg["nonlinear_study"]["alpha_sweep"]:
        J[f"NB{a:g}"] = dict(h=hd, x0=1.0, buck=True, freq=False, alpha=a)
    return J


def phase2(cfg, base):
    """Design-target sweep: from the base design, the final stages with and without buckling at
    each frequency target, each followed by the refined-mesh check."""
    sw = cfg["sweep"]
    J = {}
    for t in sw["targets"]:
        for buck, tag in ((False, "A"), (True, "B")):
            J[f"{tag}_{t:g}"] = dict(h=base["h"], x0=base["x"], buck=buck, target=t, stages=sw["polish_stages"])
    return J


def _cache_path(label, cfg, spec):
    """Checkpoint file for one optimization run; the name changes whenever its inputs change."""
    key = hashlib.sha1(pickle.dumps((cfg, {k: v for k, v in spec.items() if k not in ("x0", "resume")},
                                     np.asarray(spec["x0"], float).tobytes()))).hexdigest()[:12]
    return RES / "run_cache" / f"{label}_{key}.pkl"


def _job(args):
    label, cfg, spec = args
    from src.optimize import run
    from src.report import metrics
    path = _cache_path(label, cfg, spec)
    if spec.get("resume") and path.exists():
        print(f"[{label}] reused checkpoint {path.name}", flush=True)
        with open(path, "rb") as fh:
            return pickle.load(fh)
    cfg = copy.deepcopy(cfg)
    if spec.get("alpha") is not None:
        cfg["limits"]["buckling_min"] = spec["alpha"]
    if spec.get("target") is not None:
        cfg["limits"]["freq_design"] = spec["target"]
    h, x0, buck, freq = spec["h"], spec["x0"], spec["buck"], spec.get("freq", True)
    R = run(cfg, h, x0, buck, freq, label, log=lambda s: print(s, flush=True), stages=spec.get("stages"))
    out = dict(label=label, h=h, x0=x0 if np.ndim(x0) == 0 else "warm start", use_buckling=buck, use_freq=freq,
               alpha_req=cfg["limits"]["buckling_min"], target=cfg["limits"]["freq_design"],
               active=R["active"], x=R["x"], rho=R["final"]["rho"], history=R["history"], kkt=R["kkt"],
               mma_multipliers=R["mma_multipliers"], chosen=R["chosen"], last=R["last"], stages=R["stages"],
               seconds=R["seconds"], iterations=R["iterations"], metrics=metrics(R["model"], R["final"]),
               cfg_limits=dict(cfg["limits"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def refine(cfg, r):
    """Independent refined-mesh check: every voxel split into eight, static + modal analysis.
    Run one at a time in the main process (a 5 mm model needs several GB)."""
    from src.bracket import Bracket
    from src.report import _upsample, metrics
    cfg = copy.deepcopy(cfg)
    cfg["limits"]["freq_design"] = r["target"]
    p, beta = r["stages"][-1][:2]
    coarse, fine = Bracket(cfg, r["h"]), Bracket(cfg, r["h"] / 2)
    rf = fine.analyze(None, p, beta, grad=False, buckling=False, rho=_upsample(r["rho"], coarse.nel))
    return metrics(fine, rf)


def select(cfg, runs):
    """Lightest with-buckling sweep candidate that passes every check on the refined mesh with
    the required frequency margin and meets the smooth constraints to tol_feas_final."""
    lim, sw, opt = cfg["limits"], cfg["sweep"], cfg["optimizer"]
    rows = []
    for t in sw["targets"]:
        r = runs[f"B_{t:g}"]
        m, f = r["metrics"], r["refined"]
        checks = dict(
            refined_freq=f["freq_Hz"][0] >= lim["freq_min"] + sw["refined_margin_hz"],
            refined_stress=max(f["pnorm_MPa"]) <= cfg["material"]["yield"] / lim["safety_factor"] / 1e6,
            refined_disp=max(f["disp_mm"]) <= lim["disp_max"] * 1e3,
            smooth_constraints=max(m["g"][c] for c in m["g"] if m["g"][c] is not None) <= opt["tol_feas_final"],
            buckling=m["alpha_min"] >= lim["buckling_min"],
            connected=m["connectivity"]["passive_connected"] and m["connectivity"]["components"] == 1)
        rows.append(dict(target=t, mass=m["mass"], ok=all(checks.values()), checks=checks))
    ok = [r for r in rows if r["ok"]]
    if not ok:  # report the most conservative candidate, flagged as failing; run_all then exits non-zero
        best = max(rows, key=lambda r: runs[f"B_{r['target']:g}"]["refined"]["freq_Hz"][0])
        return best["target"], rows
    best = min(ok, key=lambda r: r["mass"])
    return best["target"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild figures and site from results/runs.pkl")
    ap.add_argument("--resume", action="store_true", help="reuse finished optimization runs from results/run_cache")
    a = ap.parse_args()
    cfg = load()
    if a.quick:
        cfg["simp"]["continuation"] = [[p, b, max(5, int(n) // 4)] for p, b, n in cfg["simp"]["continuation"]]
        cfg["sweep"]["polish_stages"] = [[p, b, max(5, int(n) // 10), *mv] for p, b, n, *mv in cfg["sweep"]["polish_stages"]]
        cfg["sweep"]["targets"] = cfg["sweep"]["targets"][::2]
    mode = "quick" if a.quick else "final"
    t0 = time.perf_counter()
    store = RES / "runs.pkl"

    if not a.report_only:
        from src import verification as V, gradcheck as GC
        RES.mkdir(exist_ok=True)
        (RES / "environment.json").write_text(json.dumps(environment(mode), indent=2))
        print("verification ...", flush=True)
        rows, conv = V.run_all(cfg)
        V.write_csv(rows, RES / "verification.csv")
        mesh_conv = V.bracket_convergence(cfg)
        write_csv(mesh_conv, RES / "mesh_convergence.csv")
        print("gradient checks ...", flush=True)
        grows, sweep = GC.directional(cfg)
        write_csv(grows, RES / "gradient_check.csv")
        write_csv(sweep, RES / "gradient_sweep.csv")
        mi = GC.method_i_check(cfg)
        write_csv(mi, RES / "gradient_method_i.csv")
        gcost = GC.cost(cfg, hs=(cfg["mesh"]["h_dev"],) if a.quick else (cfg["mesh"]["h_dev"], cfg["mesh"]["h_report"]))
        ffd = None if a.quick else GC.full_fd(cfg, cfg["mesh"]["h_dev"])
        print("optimizations ...", flush=True)
        jobs = phase1(cfg, a.quick)
        for v in jobs.values():
            v["resume"] = a.resume
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
            runs = {r["label"]: r for r in ex.map(_job, [(k, cfg, v) for k, v in jobs.items()])}
        print("design-target sweep ...", flush=True)
        jobs = phase2(cfg, runs["A_base"])
        for v in jobs.values():
            v["resume"] = a.resume
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
            runs.update({r["label"]: r for r in ex.map(_job, [(k, cfg, v) for k, v in jobs.items()])})
        print("refined-mesh checks ...", flush=True)
        for t in cfg["sweep"]["targets"]:
            runs[f"B_{t:g}"]["refined"] = refine(cfg, runs[f"B_{t:g}"])
        target, selection = select(cfg, runs)
        runs["A"], runs["B"] = runs[f"A_{target:g}"], runs[f"B_{target:g}"]
        print(f"selected design target {target:g} Hz", flush=True)
        with open(store, "wb") as fh:
            pickle.dump(dict(mode=mode, cfg=cfg, rows=rows, conv=conv, mesh_conv=mesh_conv, grows=grows, sweep=sweep,
                             method_i=mi, gcost=gcost, full_fd=ffd, runs=runs, selection=selection, target=target, t_pipeline=time.perf_counter() - t0), fh)

    from src import report
    with open(store, "rb") as fh:
        data = pickle.load(fh)
    report.build(data)
    failed = (["design selection: no sweep candidate passed every check"] if not any(r["ok"] for r in data["selection"]) else []) + \
             [r["id"] for r in data["rows"] if not r["passed"]] + \
             [f"{r['direction']}/{r['function']}" for r in data["grows"] if not r["passed"]] + \
             [r["function"] for r in data["method_i"] if not r["passed"]]
    print(f"done in {time.perf_counter() - t0:.0f} s ({data['mode']} mode); failed checks: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
