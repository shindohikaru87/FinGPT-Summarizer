# Tiny .env loader (no third-party deps)
# Loads KEY=VALUE pairs from ".env" in repo root into os.environ if not already set.
from __future__ import annotations
import os, io, pathlib

def load_env(filename: str = ".env") -> None:
    root = pathlib.Path(__file__).resolve().parents[1]  # repo root (.. from scripts/)
    env_path = root / filename
    if not env_path.exists():
        return
    for line in io.open(env_path, "r", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        # strip surrounding quotes if present
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        # don't overwrite if already set in the environment
        os.environ.setdefault(k, v)
