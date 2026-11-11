"""Regenerate every result, figure and the report site from source.

    python run_all.py           final mode: principal optimizations on the 10 mm mesh (about 1 h)
    python run_all.py --quick   pipeline check: everything on the 20 mm mesh, shortened continuation

All numbers in results/ and docs/ come from this script. Generated files are overwritten.
"""
import argparse
import copy
import csv
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


def job_list(cfg, quick):
    hd = cfg["mesh"]["h_dev"]
    hr = hd if quick else cfg["mesh"]["h_report"]
    J = {"A": (hr, 1.0, False, True, None), "B": (hr, 1.0, True, True, None)}
    for i, x0 in enumerate(cfg["optimizer"]["starts"]):
        J[f"S{i + 1}"] = (hd, x0, True, True, None)
    J["A_dev"] = (hd, 1.0, False, True, None)
    J["N"] = (hd, 1.0, False, False, None)
    J["NB"] = (hd, 1.0, True, False, None)
    for a in cfg["nonlinear_study"]["alpha_sweep"]:
        J[f"NB{a:g}"] = (hd, 1.0, True, False, a)
    return J


def _job(args):
    label, cfg, (h, x0, buck, freq, alpha) = args
    from src.optimize import run
    from src.report import metrics
    if alpha is not None:
        cfg = copy.deepcopy(cfg)
        cfg["limits"]["buckling_min"] = alpha
    R = run(cfg, h, x0, buck, freq, label, log=lambda s: print(s, flush=True))
    return dict(label=label, h=h, x0=x0, use_buckling=buck, use_freq=freq, alpha_req=cfg["limits"]["buckling_min"],
                active=R["active"], x=R["x"], rho=R["final"]["rho"], history=R["history"], kkt=R["kkt"],
                mma_multipliers=R["mma_multipliers"], chosen=R["chosen"], seconds=R["seconds"],
                iterations=R["iterations"], metrics=metrics(R["model"], R["final"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild figures and site from results/runs.pkl")
    a = ap.parse_args()
    cfg = load()
    if a.quick:
        cfg["simp"]["continuation"] = [[p, b, max(5, int(n) // 4)] for p, b, n in cfg["simp"]["continuation"]]
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
        print("optimizations ...", flush=True)
        jobs = job_list(cfg, a.quick)
        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
            runs = {r["label"]: r for r in ex.map(_job, [(k, cfg, v) for k, v in jobs.items()])}
        with open(store, "wb") as fh:
            pickle.dump(dict(mode=mode, cfg=cfg, rows=rows, conv=conv, mesh_conv=mesh_conv, grows=grows, sweep=sweep,
                             method_i=mi, gcost=gcost, runs=runs, t_pipeline=time.perf_counter() - t0), fh)

    from src import report
    with open(store, "rb") as fh:
        data = pickle.load(fh)
    report.build(data)
    failed = [r["id"] for r in data["rows"] if not r["passed"]] + \
             [f"{r['direction']}/{r['function']}" for r in data["grows"] if not r["passed"]] + \
             [r["function"] for r in data["method_i"] if not r["passed"]]
    print(f"done in {time.perf_counter() - t0:.0f} s ({data['mode']} mode); failed checks: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
