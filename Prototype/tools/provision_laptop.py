"""Explicit setup downloads, never used by inference. Record actual hashes separately from old experiments."""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
REPOS = {
    "vi": ("csukuangfj/sherpa-onnx-zipformer-vi-int8-2025-04-20", "b2745a435379992ad3f299635468db0c34918e1e", "12-avg-8"),
    "en": ("csukuangfj/sherpa-onnx-zipformer-en-2023-04-01", "34735501afc894bcee0123f4d05842ebdde30b27", "99-avg-1"),
    "ko": ("k2-fsa/sherpa-onnx-zipformer-korean-2024-06-24", "0fb4b2b5c8d3e5766121481ba911961e3649c664", "99-avg-1"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        print("download", path.relative_to(ROOT), flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        temporary.replace(path)
    return path


def hf(repo, revision, file, destination):
    return download(f"https://huggingface.co/{repo}/resolve/{revision}/{file}", destination)


def archive(name, destination):
    packed = download(RELEASE + "tts-models/" + name + ".tar.bz2", MODELS / "downloads" / (name + ".tar.bz2"))
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(packed) as tar:
        for member in tar:
            relative = Path(*Path(member.name).parts[1:])
            if not relative.parts or member.isdir():
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()) or not member.isfile():
                raise ValueError("unsafe_archive_member")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def provision(groups):
    for lang in ("vi", "en", "ko"):
        if "asr" not in groups:
            continue
        repo, revision, epoch = REPOS[lang]
        dest = MODELS / f"asr/zipformer-{lang}-int8"
        for part in ("encoder", "decoder", "joiner"):
            file = f"{part}-epoch-{epoch}" + (".onnx" if lang == "vi" and part == "decoder" else ".int8.onnx")
            hf(repo, revision, file, dest / file)
        for file in ("tokens.txt", "README.md", "test_wavs/0.wav", "test_wavs/1.wav"):
            hf(repo, revision, file, dest / file)
        if lang == "vi":
            hf(repo, revision, "test_wavs/2.wav", dest / "test_wavs/2.wav")
            decoder = dest / f"decoder-epoch-{epoch}.int8.onnx"
            if not decoder.exists():
                from onnxruntime.quantization import QuantType, quantize_dynamic
                quantize_dynamic(str(dest / f"decoder-epoch-{epoch}.onnx"), str(decoder), weight_type=QuantType.QInt8)
    if "tts" in groups:
        for name in ("vits-piper-en_US-ljspeech-medium",):
            archive(name, MODELS / "tts" / name)
        archive("sherpa-onnx-supertonic-3-tts-int8-2026-05-11", MODELS / "tts/supertonic-multilingual")
    if "baseline" in groups:
        # Same checkpoint family, a public ONNX export. These are not the old audit's exact NMT artifacts.
        repo, revision = "Xenova/opus-mt-vi-en", "541dfd72ec07e8f1563795e0224408ad8d729e80"
        dest = MODELS / "nmt/vi-en-int8-arm64"
        for file in ("config.json", "source.spm", "target.spm", "vocab.json", "README.md"):
            hf(repo, revision, file, dest / file)
        for file in ("encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"):
            hf(repo, revision, "onnx/" + file, dest / file)
    if "vad" in groups:
        download(RELEASE + "asr-models/silero_vad.onnx", MODELS / "silero_vad.onnx")
    if "branch-b" in groups:
        import swift_f0
        sys.path.insert(0, str(ROOT / "src"))
        from tonebridge.branch_b_graph import build
        dest = MODELS / "branch_b"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(swift_f0.__file__).parent / "model.onnx", dest / "swiftf0_model.onnx")
        if not (dest / "band_feats.onnx").exists():
            build(str(dest / "band_feats.onnx"))
    paths = [p for p in MODELS.rglob("*") if p.is_file() and "downloads" not in p.parts and "vits-piper-vi_VN-vivos-x_low" not in p.parts and not p.name.endswith(".part")]
    rows = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(paths)]
    (ROOT / "configs/laptop_assets_manifest.json").write_text(json.dumps({"note": "Current laptop restoration; distinct from historical runtime_files.json", "asr_sources": REPOS,
                                                                       "files": rows}, indent=2), encoding="utf-8")
    print("manifest", len(rows), "files", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", choices=["asr", "tts", "baseline", "vad", "branch-b"], default=["asr", "tts", "baseline", "vad", "branch-b"])
    provision(ap.parse_args().groups)
