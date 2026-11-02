from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text())
