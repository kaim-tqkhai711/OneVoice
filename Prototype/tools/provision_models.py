"""Provision official model packs without changing the runtime trust manifest.

Run with the benchmark/export virtualenv. Derived graphs are rebuilt locally;
they are accepted ONLY if their bytes match the original manifest. A compatible
model with a different hash is not a replacement for the pinned artifact.
Use --from-directory for an original offline pack (paths relative to models/).
No upstream Python code or pickle is executed; the Marian checkpoint is loaded
by Transformers with torch's weights_only safeguard.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tonebridge.assets import selected_assets, verify_files
from tonebridge.config import PipelineConfig, sha256_file

ASR_REPO = "csukuangfj/sherpa-onnx-zipformer-vi-int8-2025-04-20"
ASR_REVISION = "b2745a435379992ad3f299635468db0c34918e1e"
ASR_DECODER_REVISION = "e827965a37aab92a4455566fac49c0e80a23afef"
TTS_REPO = "csukuangfj/vits-piper-en_US-ljspeech-medium"
TTS_REVISION = "34f390d13b9edbaa30c339127de0f9c4c5948c90"
NMT_REPO = "Helsinki-NLP/opus-mt-vi-en"
NMT_REVISION = "c8d2853e77f5fae31124d993e0b35176b1c8914e"
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SOURCES = {
    "vad": RELEASE + "/asr-models/silero_vad.onnx",
    "asr": RELEASE + "/asr-models/sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2",
    "asr_decoder": ("https://huggingface.co/zzasdf/viet_iter3_pseudo_label/resolve/"
                    + ASR_DECODER_REVISION + "/exp/decoder-epoch-12-avg-8.int8.onnx"),
    "tts": RELEASE + "/tts-models/vits-piper-en_US-ljspeech-medium.tar.bz2",
    "nmt": f"https://huggingface.co/{NMT_REPO}/tree/{NMT_REVISION}",
    "swift_f0": "https://pypi.org/project/swift-f0/0.3.0/",
    "band_features": "src/tonebridge/branch_b_graph.py",
    "piper_origin": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ljspeech/medium",
    "export_tool": "https://github.com/huggingface/optimum-onnx",
}


class ProvisionError(RuntimeError):
    """A source, integrity or installation error; never permits fallback."""


def checked(path: Path, row: dict) -> None:
    """Validate bytes and digest, deleting only the rejected staging file."""
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_size != row["bytes"] or actual_hash != row["sha256"]:
        path.unlink()
        raise ProvisionError(
            f"integrity mismatch: {Path(row['path']).name}; expected bytes={row['bytes']} "
            f"sha256={row['sha256']}; actual bytes={actual_size} sha256={actual_hash}")


def download(url: str, path: Path, limit: int, *, timeout: float = 60) -> None:
    """Stream bounded HTTPS downloads; never leave a failed/partial file."""
    if not url.startswith("https://"):
        raise ProvisionError("only HTTPS sources are allowed")
    if path.exists():
        raise ProvisionError("download destination already exists")
    start = time.monotonic()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "OneVoice-model-provision/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response, path.open("xb") as output:
            if not response.geturl().startswith("https://"):
                raise ProvisionError("HTTPS downgrade rejected")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > limit:
                raise ProvisionError("download exceeds size limit")
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > limit or time.monotonic() - start > 900:
                    raise ProvisionError("download exceeds size/time limit")
                output.write(block)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def extract_selected(archive: Path, prefix: str, wanted: dict[str, dict], stage: Path) -> None:
    """Extract a whitelist, rejecting traversal, links and duplicate members.

    No extractall(): unlisted README/scripts/audio are not installed. Output
    sizes are bounded by the manifest before any member is copied.
    """
    seen = set()
    with tarfile.open(archive, "r:bz2") as pack:
        for member in pack:
            name = member.name
            parts = PurePosixPath(name).parts
            if (name.startswith("/") or "\\" in name or ":" in name or ".." in parts
                    or member.issym() or member.islnk() or not (member.isfile() or member.isdir())):
                raise ProvisionError(f"unsafe archive member: {name}")
            if name in seen:
                raise ProvisionError(f"duplicate archive member: {name}")
            seen.add(name)
            if not member.isfile() or not name.startswith(prefix + "/"):
                continue
            relative = name[len(prefix) + 1:]
            row = wanted.get(relative)
            if row is None:
                continue
            if member.size != row["bytes"]:
                raise ProvisionError(f"archive size mismatch: {relative}")
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with pack.extractfile(member) as source, path.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            checked(path, row)


def install_group(stage: Path, entries: list[dict], base: Path) -> int:
    """Check the complete group first, then atomically install missing files.

    Existing bad files are deleted and rejected (never silently repaired).
    Roll back newly installed paths if an installation error occurs.
    """
    pending = []
    for row in entries:
        target = Path(row["path"])
        if target.exists():
            issues = verify_files([row])
            if issues:
                target.unlink()
                raise ProvisionError(issues[0] + "; rejected file deleted")
            continue
        source = stage / target.relative_to(base)
        if not source.is_file():
            raise ProvisionError(f"source does not contain pinned asset: {target.name}")
        checked(source, row)
        pending.append((source, target))
    installed = []
    try:
        for source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create protects an independently written target. Staging
            # and targets are on the same volume; hard link exposes complete bytes.
            os.link(source, target)
            installed.append(target)
    except BaseException:
        for target in installed:
            target.unlink(missing_ok=True)
        raise
    return len(installed)


def hf_url(repo: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def swift_wheel_model(stage: Path, row: dict) -> None:
    """Copy the exact installed 0.3.0 wheel model, or fetch its PyPI wheel.

    Downloading the model does not install dependencies; use the runtime
    requirements before check_runtime. Only one bounded member is extracted.
    """
    try:
        dist = metadata.distribution("swift-f0")
    except metadata.PackageNotFoundError:
        dist = None
    output = stage / Path(row["path"]).name
    if dist is not None and dist.version == "0.3.0":
        bundled = [dist.locate_file(f) for f in dist.files or []
                   if str(f).replace("\\", "/") == "swift_f0/model.onnx"]
        if len(bundled) != 1:
            raise ProvisionError("swift-f0 wheel does not contain exactly one bundled model")
        shutil.copyfile(bundled[0], output)
        return
    info = stage / "pypi.json"
    download("https://pypi.org/pypi/swift-f0/0.3.0/json", info, 1024 * 1024)
    releases = json.loads(info.read_text(encoding="utf8"))["urls"]
    wheels = [f for f in releases if f["filename"].endswith("py3-none-any.whl")]
    if len(wheels) != 1:
        raise ProvisionError("expected one platform independent swift-f0 0.3.0 wheel")
    release = wheels[0]
    if not release["url"].startswith("https://files.pythonhosted.org/"):
        raise ProvisionError("unexpected PyPI file host")
    wheel = stage / "swift.whl"
    download(release["url"], wheel, release["size"])
    checked(wheel, {"path": str(wheel), "bytes": release["size"], "sha256": release["digests"]["sha256"]})
    with zipfile.ZipFile(wheel) as pack:
        matches = [m for m in pack.infolist() if m.filename == "swift_f0/model.onnx"]
        if len(matches) != 1 or matches[0].file_size != row["bytes"]:
            raise ProvisionError("SwiftF0 wheel member missing/duplicate/size mismatch")
        with pack.open(matches[0]) as source, output.open("xb") as dest:
            shutil.copyfileobj(source, dest, 1024 * 1024)


def quantize_nmt(fp32: Path, stage: Path) -> None:
    """Apply exactly the dynamic INT8 arm64 settings used by bench_hop.py."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    quant = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    for name in ("encoder_model.onnx", "decoder_model_merged.onnx"):
        ORTQuantizer.from_pretrained(fp32, file_name=name).quantize(
            save_dir=stage, quantization_config=quant)


def _build_nmt(stage: Path, quantizer_path: Path | None = None) -> None:
    """Export official Marian, then apply the repo's dynamic arm64 recipe.

    Toolchain is pinned by requirements-bench; hashes, not export success, are
    authoritative. Original artifacts may require the original export toolchain.
    Runtime never calls this function or downloads anything.
    """
    from transformers import MarianTokenizer
    from optimum.exporters.onnx import main_export

    upstream = stage / "upstream"
    fp32 = stage / "fp32"
    upstream.mkdir()
    for name, size in (("pytorch_model.bin", 288977385), ("config.json", 1394),
                       ("source.spm", 756168), ("target.spm", 808853),
                       ("vocab.json", 1193597), ("tokenizer_config.json", 44)):
        download(hf_url(NMT_REPO, NMT_REVISION, name), upstream / name, size)
    weights = upstream / "pytorch_model.bin"
    checked(weights, {"path": str(weights), "bytes": 288977385,
                      "sha256": "b824e712e1668a5387d897cdd55f58ef45431bd61b14298d0426650cda4de58d"})
    # Offline local source and trust_remote_code=False; no repository scripts.
    main_export(str(upstream), fp32, task="text2text-generation-with-past", device="cpu",
                local_files_only=True, trust_remote_code=False, do_validation=False)
    MarianTokenizer.from_pretrained(upstream, local_files_only=True).save_pretrained(fp32)
    if quantizer_path is None:
        quantize_nmt(fp32, stage)
    else:
        # Export/quantizer experiments are isolated from runtime ORT sessions.
        # Only an explicitly supplied local tooling path is executable here.
        code = ("import sys,runpy; from pathlib import Path; "
                "sys.path.insert(0,sys.argv[1]); "
                "m=runpy.run_path(sys.argv[2],run_name='provision_library'); "
                "m['quantize_nmt'](Path(sys.argv[3]),Path(sys.argv[4]))")
        subprocess.run([sys.executable, "-c", code, str(quantizer_path), str(Path(__file__).resolve()),
                        str(fp32), str(stage)], check=True, timeout=600)
    for name in ("source.spm", "target.spm", "vocab.json", "config.json"):
        shutil.copyfile(fp32 / name, stage / name)


def build_nmt(stage: Path, quantizer_path: Path | None = None) -> None:
    """Freeze Python hash seed before importing the exporter in a child process.

    Optimum's merge deduplication uses set.pop() for initializer names. Freezing
    the seed avoids varying names on retries, but cannot infer the original
    artifact's unrecorded Python/hash seed. The manifest still decides acceptance.
    """
    code = ("import sys,runpy; from pathlib import Path; "
            "m=runpy.run_path(sys.argv[1],run_name='provision_library'); "
            "m['_build_nmt'](Path(sys.argv[2]),Path(sys.argv[3]) if sys.argv[3] else None)")
    subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()), str(stage),
                    str(quantizer_path) if quantizer_path is not None else ""],
                   env={**os.environ, "PYTHONHASHSEED": "0"}, check=True, timeout=1800)


def provision_group(key: str, entries: list[dict], base: Path, stage: Path,
                    local: Path | None, quantizer_path: Path | None = None) -> int:
    """Prepare one pack, using a supplied original directory when available."""
    if not verify_files(entries):
        return 0
    if local is not None:
        for row in entries:
            target = Path(row["path"])
            candidate = local / target.relative_to(base)
            if candidate.is_file():
                dest = stage / target.relative_to(base)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, dest)
        return install_group(stage, entries, base)
    wanted = {str(Path(r["path"]).relative_to(base)).replace("\\", "/"): r for r in entries}
    if key in ("asr", "tts"):
        archive = stage / "pack.tar.bz2"
        download(SOURCES[key], archive, 70_000_000)
        prefix = "sherpa-onnx-zipformer-vi-int8-2025-04-20" if key == "asr" else "vits-piper-en_US-ljspeech-medium"
        extract_selected(archive, prefix, wanted, stage)
        if key == "asr":
            name = "decoder-epoch-12-avg-8.int8.onnx"
            if name in wanted and not (stage / name).exists() and not Path(wanted[name]["path"]).exists():
                download(SOURCES["asr_decoder"], stage / name, wanted[name]["bytes"])
    elif key == "vad":
        row = entries[0]
        download(SOURCES[key], stage / Path(row["path"]).name, row["bytes"])
    elif key == "swift_f0":
        swift_wheel_model(stage, entries[0])
    elif key == "band_features":
        from tonebridge.branch_b_graph import build
        build(str(stage / Path(entries[0]["path"]).name))
    elif key == "nmt":
        build_nmt(stage, quantizer_path)
        # Retain independently verified tokenizer/encoder assets. A missing
        # decoder still blocks both preflights and all inference sessions.
        auxiliary = [r for r in entries if "decoder_model" not in Path(r["path"]).name]
        installed = install_group(stage, auxiliary, base)
        decoder = [r for r in entries if r not in auxiliary]
        return installed + install_group(stage, decoder, base)
    else:
        raise ProvisionError(f"no official provisioning recipe: {key}; supply original --from-directory")
    return install_group(stage, entries, base)


@contextmanager
def exclusive_lock(root: Path):
    """Prevent concurrent provisioners; never steal a stale lock implicitly."""
    lock = root / ".provision-models.lock"
    try:
        with lock.open("x", encoding="utf8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ProvisionError(f"provisioner lock exists: {lock}") from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run(cfg: PipelineConfig, mode: str, local: Path | None = None,
        quantizer_path: Path | None = None) -> dict:
    """Provision active assets only; return auditable metadata and final checks."""
    if mode not in ("branch-a-bench", "full"):
        raise ProvisionError("provisioning requires branch-a-bench or full")
    entries = selected_assets(cfg, mode)
    models = (cfg.root / "models").resolve()
    for row in entries:
        p = Path(row["path"])
        if not p.is_relative_to(models):
            raise ProvisionError(f"asset outside models/: {p}")
        if p.exists() and verify_files([row]):
            p.unlink()
            raise ProvisionError(f"existing asset rejected and deleted: {p}")
    groups = []
    for key, field in (("vad", "vad"), ("asr", "asr_vi"), ("tts", "tts_en"),
                       ("swift_f0", "swift_f0"), ("band_features", "band_features"), ("nmt", "nmt_vi_en")):
        model = cfg.model_path(field)
        base = model.parent if model.suffix == ".onnx" else model
        rows = [r for r in entries if Path(r["path"]) == model or (model.suffix != ".onnx" and Path(r["path"]).is_relative_to(model))]
        if rows:
            groups.append((key, base, rows))
    covered = {r["path"] for _, _, rows in groups for r in rows}
    extras = [r for r in entries if r["path"] not in covered]
    if extras:
        groups.append(("custom", models, extras))
    report = {"mode": mode, "assets_selected": len(entries), "sources": SOURCES,
              "manifest_sha256": sha256_file(cfg.resolve_path(cfg.asset_manifest)), "groups": [], "ok": False,
              "nmt_export_hash_seed": "0", "toolchain": {n: metadata.version(n) for n in
                  ("numpy", "onnxruntime")}}
    if quantizer_path is not None:
        report["nmt_quantizer_path"] = str(quantizer_path)
        report["nmt_quantizer_version"] = next((d.version for d in metadata.distributions(path=[str(quantizer_path)])
            if d.metadata["Name"] == "onnxruntime"), None)
    with exclusive_lock(cfg.root):
        for key, base, rows in groups:
            print(f"provision {key}: {len(rows)} pinned files", file=sys.stderr, flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix=".provision-", dir=cfg.root) as tmp:
                    source_dir = local / base.relative_to(models) if local is not None else None
                    count = provision_group(key, rows, base, Path(tmp), source_dir, quantizer_path)
                report["groups"].append({"group": key, "verified": len(rows), "installed": count})
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"] = verify_files(entries)
                report["assets_verified"] = len(entries) - len(report["errors"])
                return report
        report["errors"] = verify_files(entries)
        report["assets_verified"] = len(entries) - len(report["errors"])
        report["ok"] = not report["errors"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("branch-a-bench", "full"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--from-directory", type=Path, help="original offline models/ directory; identical pins required")
    parser.add_argument("--nmt-quantizer-path", type=Path,
                        help="isolated original ORT tooling directory; never modifies runtime dependencies")
    parser.add_argument("--out", type=Path, help="save provisioning metadata (no model bytes)")
    args = parser.parse_args(argv)
    try:
        cfg = PipelineConfig.load(args.config)
        if args.from_directory and not args.from_directory.is_dir():
            raise ProvisionError("--from-directory must be an existing original models/ directory")
        if args.nmt_quantizer_path and not (args.nmt_quantizer_path / "onnxruntime/__init__.py").is_file():
            raise ProvisionError("--nmt-quantizer-path must contain the explicitly selected onnxruntime package")
        result = run(cfg, args.mode, args.from_directory.resolve() if args.from_directory else None,
                     args.nmt_quantizer_path.resolve() if args.nmt_quantizer_path else None)
    except Exception as exc:
        result = {"ok": False, "mode": args.mode, "error": f"{type(exc).__name__}: {exc}"}
    rendered = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf8")
    # Full error inventory is saved in --out; stdout remains concise.
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
