"""Config loading and path helpers."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def path(cfg: dict, *parts) -> Path:
    """Resolve a path relative to the project root (folder of config.yaml)."""
    return Path(cfg["_root"], *parts)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
