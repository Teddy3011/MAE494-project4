"""One command: regenerate every result. Usage: python run_all.py"""
import json
import platform
import sys
from importlib.metadata import version

from src import verification
from src.config import ROOT, load


def write_environment():
    env = dict(python=sys.version, platform=platform.platform(),
               packages={p: version(p) for p in ["numpy", "scipy", "jax", "jaxlib", "PyYAML"]})
    path = ROOT / "results" / "environment.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    cfg = load()
    write_environment()
    rows = verification.run_all(cfg)
    print("wrote", verification.write_csv(rows))
