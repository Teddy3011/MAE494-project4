"""Final-stage convergence study (separate from run_all.py; about 50 min on 4 processes).

Starts every run from the same design: the 107 Hz with-buckling design of commit 4cfd53a
(studies/start_B107_4cfd53a.npy), on the 10 mm mesh, with the engineering problem unchanged.
Each run changes one final-stage setting. Writes results/convergence_study.csv.

    python studies/convergence_study.py            run all cases, then summarize
    python studies/convergence_study.py --summarize DIR   rebuild the CSV from saved runs in DIR
"""
import csv
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES = {  # label: (final stages [p, beta, max iterations, move], description)
    "E1_m015": ([[3.0, 8.0, 150, 0.015]], "beta 8, move 0.015"),
    "E2_m010": ([[3.0, 8.0, 150, 0.010]], "beta 8, move 0.010"),
    "E3_b16": ([[3.0, 16.0, 150, 0.015]], "beta 16, move 0.015"),
    "E4_b16_32": ([[3.0, 16.0, 75, 0.015], [3.0, 32.0, 75, 0.015]], "beta 16 then 32, move 0.015"),
}


def job(args):
    label, stages, outdir = args
    from src.bracket import Bracket
    from src.config import load
    from src.optimize import run
    from src.report import _upsample, metrics
    cfg = load()
    cfg["limits"]["freq_design"] = 107.0
    x0 = np.load(ROOT / "studies" / "start_B107_4cfd53a.npy")
    R = run(cfg, 0.01, x0, True, True, label, log=lambda s: print(s, flush=True), stages=stages)
    p, beta = stages[-1][:2]
    m10 = metrics(R["model"], R["final"])
    fine = Bracket(cfg, 0.005)
    r5 = fine.analyze(None, p, beta, grad=False, buckling=False, rho=_upsample(R["final"]["rho"], R["model"].nel))
    out = dict(label=label, stages=stages, iterations=R["iterations"], seconds=R["seconds"], chosen=R["chosen"],
               last=dict(mass=R["last"]["mass"], max_g=R["last"]["max_g"]), kkt=R["kkt"], mma=R["mma_multipliers"],
               mass=m10["mass"], f10=m10["freq_Hz"][0], f5=float(r5["freq"][0]), alpha=m10["alpha_min"],
               pn5=float(max(r5["pnorm"]) / 1e6), disp5=float(max(np.linalg.norm(r5["disp"], axis=1)) * 1e3),
               maxg=float(max(R["final"]["g"][i] for i in R["active"])), gray=m10["grayness"], conn=m10["connectivity"],
               x=R["x"], hist=[(h["iter"], h["mass"], h["freq"], h["change"]) for h in R["history"]])
    pickle.dump(out, open(Path(outdir) / f"{label}.pkl", "wb"))
    return label


def summarize(outdir):
    rows = []
    for label, (_, desc) in CASES.items():
        r = pickle.load(open(Path(outdir) / f"{label}.pkl", "rb"))
        rows.append(dict(case=label, setting=desc, iterations=r["iterations"], seconds=round(r["seconds"]),
                         reported=r["chosen"], mass_kg=r["mass"], f1_10mm_Hz=r["f10"], f1_5mm_Hz=r["f5"],
                         alpha_min=r["alpha"], max_g=r["maxg"], stationarity=r["kkt"]["stationarity"],
                         bound_sign=r["kkt"]["bound_sign_violation"], complementarity=r["kkt"]["complementarity"],
                         grayness=r["gray"], components=r["conn"]["components"],
                         last_iterate_mass_kg=r["last"]["mass"], last_iterate_max_g=r["last"]["max_g"]))
    path = ROOT / "results" / "convergence_study.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1, default=float))
    return path


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
        sys.exit()
    outdir = ROOT / "results" / "convergence_runs"
    outdir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(len(CASES)) as ex:
        list(ex.map(job, [(k, s, str(outdir)) for k, (s, _) in CASES.items()]))
    summarize(outdir)
