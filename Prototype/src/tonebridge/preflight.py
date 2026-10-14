"""Environment diagnostics usable even before inference dependencies are installed."""
from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMPORTS = {"sherpa-onnx": "sherpa_onnx", "swift-f0": "swift_f0", "soundfile": "soundfile",
           "optimum": "optimum.onnxruntime", "optimum-onnx": "optimum.onnxruntime"}
BASE = {"numpy", "scipy", "pydantic", "psutil"}


def requirements(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    """Expand local -r includes; require exact pins for each direct dependency."""
    seen = set() if seen is None else seen
    path = path.resolve(strict=True)
    if path in seen:
        raise ValueError(f"cyclic requirements include: {path}")
    seen.add(path)
    pins = {}
    for raw in path.read_text(encoding="utf8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            pins.update(requirements(path.parent / line[3:].strip(), seen))
            continue
        match = re.fullmatch(r"([\w-]+)(?:\[[\w,-]+\])?==([^\s;]+)", line)
        if not match:
            raise ValueError(f"unsupported/unpinned requirement: {line}")
        name, version = match.groups()
        name = name.lower().replace("_", "-")
        if name in pins and pins[name] != version:
            raise ValueError(f"conflicting pins: {name}")
        pins[name] = version
    seen.remove(path)
    return pins


def check_dependencies(mode: str, profile: str = "runtime", *, strict_versions: bool = True,
                       project_root: Path = ROOT) -> dict:
    """Check pins and actual imports (including DLL failures), without models."""
    pins = requirements(project_root / f"requirements-{profile}.txt")
    if profile == "runtime":
        if mode == "stub":
            pins = {n: v for n, v in pins.items() if n in BASE}
        elif mode == "branch-a-bench":
            pins.pop("swift-f0", None)
    rows, errors = [], []
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        errors.append("supported Python versions: 3.11-3.13")
    for name, expected in sorted(pins.items()):
        row = {"distribution": name, "expected": expected, "installed": None, "import_ok": False}
        try:
            row["installed"] = importlib.metadata.version(name)
            importlib.import_module(IMPORTS.get(name, name.replace("-", "_")))
            row["import_ok"] = True
            # CPU/CUDA local version tags do not alter the public Torch version.
            if strict_versions and row["installed"].split("+")[0] != expected.split("+")[0]:
                errors.append(f"version mismatch: {name} installed={row['installed']} expected={expected}")
        except Exception as exc:
            errors.append(f"dependency unavailable: {name} ({type(exc).__name__}: {exc})")
        rows.append(row)
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "profile": profile, "dependencies": rows, "errors": errors}


def check_runtime(config_path: Path, mode: str, profile: str = "runtime", *, strict_versions: bool = True) -> dict:
    """Collect configuration, dependency and all selected asset errors."""
    report = {"mode": mode, "ok": False, "errors": [], "assets_checked": 0}
    environment = check_dependencies(mode, profile, strict_versions=strict_versions)
    report.update(environment)
    try:
        from tonebridge.assets import selected_assets, verify_files
        from tonebridge.config import PipelineConfig

        cfg = PipelineConfig.load(config_path)
        effective = cfg.prepare(mode, verify_assets=False)
        entries = selected_assets(effective, mode)
        report["assets_checked"] = len(entries)
        report["config_hash"] = effective.config_hash()
        report["errors"].extend(verify_files(entries))
    except Exception as exc:
        report["errors"].append(f"configuration preflight: {type(exc).__name__}: {exc}")
    report["ok"] = not report["errors"]
    return report
