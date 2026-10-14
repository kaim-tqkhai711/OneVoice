"""Offline provisioning fault tests: no network/model replacements in fixtures."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile
import json

import pytest

from tonebridge.assets import selected_assets
from tonebridge.config import PipelineConfig, ROOT

spec = importlib.util.spec_from_file_location("provision_models", ROOT / "tools/provision_models.py")
provision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provision)


def row(path, payload=b"verified"):
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.mark.parametrize("mode,count", [("branch-a-bench", 368), ("full", 370)])
def test_modes_use_runtime_selection_and_leave_manifest_unchanged(mode, count):
    cfg = PipelineConfig.load(ROOT / "configs/pipeline.json")
    before = cfg.resolve_path(cfg.asset_manifest).read_bytes()
    assert len(selected_assets(cfg, mode)) == count
    assert cfg.resolve_path(cfg.asset_manifest).read_bytes() == before


@pytest.mark.parametrize("bad", [b"short", b"corrupt!"])
def test_wrong_size_or_hash_is_deleted(tmp_path, bad):
    staged = tmp_path / "staged.bin"
    staged.write_bytes(bad)
    with pytest.raises(provision.ProvisionError, match="integrity mismatch"):
        provision.checked(staged, row(tmp_path / "target.bin"))
    assert not staged.exists()


def test_group_validates_every_file_before_install_and_is_idempotent(tmp_path):
    stage, target = tmp_path / "stage", tmp_path / "models"
    stage.mkdir()
    rows = [row(target / name) for name in ("one.bin", "two.bin")]
    (stage / "one.bin").write_bytes(b"verified")
    (stage / "two.bin").write_bytes(b"corrupt!")
    with pytest.raises(provision.ProvisionError):
        provision.install_group(stage, rows, target)
    assert not target.exists()
    assert not (stage / "two.bin").exists()
    (stage / "two.bin").write_bytes(b"verified")
    assert provision.install_group(stage, rows, target) == 2
    assert provision.install_group(stage, rows, target) == 0
    assert (target / "one.bin").read_bytes() == b"verified"


def test_install_error_rolls_back_only_new_paths(tmp_path, monkeypatch):
    stage, target = tmp_path / "stage", tmp_path / "models"
    stage.mkdir(); target.mkdir()
    rows = [row(target / name) for name in ("existing", "one", "two")]
    (target / "existing").write_bytes(b"verified")
    for name in ("one", "two"):
        (stage / name).write_bytes(b"verified")
    link = provision.os.link
    def fail_second(source, destination):
        if destination.name == "two":
            raise OSError("disk failure")
        link(source, destination)
    monkeypatch.setattr(provision.os, "link", fail_second)
    with pytest.raises(OSError):
        provision.install_group(stage, rows, target)
    assert sorted(p.name for p in target.iterdir()) == ["existing"]


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/absolute", "file"),
    ("C:/drive", "file"), ("pack/../escape", "file"), ("pack\\escape", "file"),
    ("pack/link", "symlink"), ("pack/link", "hardlink")])
def test_archive_rejects_traversal_and_links(tmp_path, name, kind):
    archive = tmp_path / "pack.tar.bz2"
    with tarfile.open(archive, "w:bz2") as pack:
        member = tarfile.TarInfo(name)
        if kind != "file":
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "../outside"
        pack.addfile(member)
    with pytest.raises(provision.ProvisionError, match="unsafe archive"):
        provision.extract_selected(archive, "pack", {}, tmp_path)


def test_archive_whitelist_and_duplicate_rejection(tmp_path):
    archive = tmp_path / "pack.tar.bz2"
    with tarfile.open(archive, "w:bz2") as pack:
        for name, value in [("pack/model.bin", b"verified"), ("pack/unlisted.py", b"ignored")]:
            member = tarfile.TarInfo(name); member.size = len(value)
            pack.addfile(member, io.BytesIO(value))
    stage = tmp_path / "stage"; stage.mkdir()
    provision.extract_selected(archive, "pack", {"model.bin": row(stage / "model.bin")}, stage)
    assert sorted(p.name for p in stage.iterdir()) == ["model.bin"]
    with tarfile.open(archive, "w:bz2") as pack:
        for _ in range(2):
            pack.addfile(tarfile.TarInfo("pack/duplicate"))
    with pytest.raises(provision.ProvisionError, match="duplicate"):
        provision.extract_selected(archive, "pack", {}, stage)


@pytest.mark.parametrize("failure", ["oversize", "io", "downgrade"])
def test_failed_download_does_not_leave_partial(tmp_path, monkeypatch, failure):
    class Response(io.BytesIO):
        headers = {}
        def geturl(self):
            return "http://invalid" if failure == "downgrade" else "https://official"
        def read(self, size):
            if failure == "io":
                raise OSError("broken connection")
            return super().read(size)
    monkeypatch.setattr(provision.urllib.request, "urlopen", lambda *a, **kw: Response(b"too long"))
    path = tmp_path / "partial"
    with pytest.raises((provision.ProvisionError, OSError)):
        provision.download("https://official", path, 3)
    assert not path.exists()


def test_lock_is_exclusive_and_released_on_error(tmp_path):
    with pytest.raises(RuntimeError, match="failure"):
        with provision.exclusive_lock(tmp_path):
            with pytest.raises(provision.ProvisionError, match="lock exists"):
                with provision.exclusive_lock(tmp_path):
                    pytest.fail("second writer acquired lock")
            raise RuntimeError("failure")
    assert not (tmp_path / ".provision-models.lock").exists()


def test_local_pack_uses_models_relative_paths(tmp_path):
    base = tmp_path / "destination/models/branch_b"
    local = tmp_path / "source/models/branch_b"; local.mkdir(parents=True)
    (local / "model.onnx").write_bytes(b"verified")
    stage = tmp_path / "stage"; stage.mkdir()
    rows = [row(base / "model.onnx")]
    assert provision.provision_group("swift_f0", rows, base, stage, local) == 1
    assert (base / "model.onnx").read_bytes() == b"verified"


def test_download_never_deletes_an_existing_file(tmp_path):
    existing = tmp_path / "existing"
    existing.write_bytes(b"original")
    with pytest.raises(provision.ProvisionError, match="already exists"):
        provision.download("https://official", existing, 8)
    assert existing.read_bytes() == b"original"


def test_existing_invalid_asset_is_deleted_without_overwriting_good_file(tmp_path):
    stage, target = tmp_path / "stage", tmp_path / "models"
    stage.mkdir(); target.mkdir()
    (target / "good").write_bytes(b"verified")
    (target / "bad").write_bytes(b"corrupt!")
    rows = [row(target / name) for name in ("good", "bad")]
    with pytest.raises(provision.ProvisionError, match="rejected file deleted"):
        provision.install_group(stage, rows, target)
    assert (target / "good").read_bytes() == b"verified"
    assert not (target / "bad").exists()


def test_swift_wheel_fallback_extracts_only_pinned_model(tmp_path, monkeypatch):
    def missing(*args):
        raise provision.metadata.PackageNotFoundError("swift-f0")
    monkeypatch.setattr(provision.metadata, "distribution", missing)
    wheel = tmp_path / "upstream.whl"
    with zipfile.ZipFile(wheel, "w") as pack:
        pack.writestr("swift_f0/model.onnx", b"verified")
        pack.writestr("../evil.py", b"never extracted")
    wheel_bytes = wheel.read_bytes()
    def fetch(url, destination, limit):
        if destination.name == "pypi.json":
            destination.write_text(json.dumps({"urls": [{"filename": "swift_f0-0.3.0-py3-none-any.whl",
                "url": "https://files.pythonhosted.org/wheel", "size": len(wheel_bytes),
                "digests": {"sha256": hashlib.sha256(wheel_bytes).hexdigest()}}]}))
        else:
            destination.write_bytes(wheel_bytes)
    monkeypatch.setattr(provision, "download", fetch)
    stage = tmp_path / "stage"; stage.mkdir()
    r = row(stage / "swiftf0_model.onnx")
    provision.swift_wheel_model(stage, r)
    provision.checked(stage / "swiftf0_model.onnx", r)
    assert not (tmp_path / "evil.py").exists()


def test_nmt_failure_retains_only_verified_assets(tmp_path, monkeypatch):
    stage, target = tmp_path / "stage", tmp_path / "models/nmt"
    stage.mkdir()
    rows = [row(target / name) for name in ("encoder.onnx", "config.json", "decoder_model.onnx")]
    def build(stage, quantizer_path):
        for r in rows:
            (stage / Path(r["path"]).name).write_bytes(
                b"corrupt!" if "decoder" in r["path"] else b"verified")
    monkeypatch.setattr(provision, "build_nmt", build)
    with pytest.raises(provision.ProvisionError, match="integrity mismatch"):
        provision.provision_group("nmt", rows, target, stage, None)
    assert sorted(p.name for p in target.iterdir()) == ["config.json", "encoder.onnx"]
    assert not (stage / "decoder_model.onnx").exists()


def test_asr_missing_decoder_downloads_official_zzasdf_file(tmp_path, monkeypatch):
    stage, target = tmp_path / "stage", tmp_path / "models/asr"
    stage.mkdir()
    name = "decoder-epoch-12-avg-8.int8.onnx"
    seen = []
    def fetch(url, destination, limit):
        seen.append(url)
        if destination.name == "pack.tar.bz2":
            with tarfile.open(destination, "w:bz2"):
                pass
        else:
            destination.write_bytes(b"verified")
    monkeypatch.setattr(provision, "download", fetch)
    assert provision.provision_group("asr", [row(target / name)], target, stage, None) == 1
    assert seen[-1] == provision.SOURCES["asr_decoder"]
    assert "zzasdf/viet_iter3_pseudo_label/resolve/" in seen[-1]


@pytest.mark.parametrize("create", [False, True])
def test_invalid_explicit_quantizer_path_fails_before_provisioning(tmp_path, monkeypatch, capsys, create):
    tooling = tmp_path / "original-ort"
    if create:
        tooling.mkdir()
    def forbidden(*args):
        pytest.fail("invalid tooling path triggered provisioning/fallback")
    monkeypatch.setattr(provision, "run", forbidden)
    assert provision.main(["--mode", "branch-a-bench", "--nmt-quantizer-path", str(tooling)]) == 1
    assert "must contain the explicitly selected" in capsys.readouterr().out
