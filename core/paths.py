import sys
from pathlib import Path


def project_root():
    if getattr(sys, "_MEIPASS", None):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[1]


def asset_path(name):
    root = project_root()
    direct = root / name
    if direct.exists():
        return direct
    in_assets = root / "assets" / name
    if in_assets.exists():
        return in_assets
    return direct