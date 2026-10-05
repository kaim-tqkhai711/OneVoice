"""Evaluation sets. Each yields (utt_id, wav_path, reference_text, tag)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[3]


def fleurs_vi(split: str = "test", limit: int | None = None) -> Iterator[tuple[str, Path, str, str]]:
    """google/fleurs vi_vn, CC-BY-4.0. Reference = FLEURS normalized transcription (tsv column 4)."""
    base = ROOT / "data/fleurs_vi/data/vi_vn"
    with open(base / f"{split}.tsv", encoding="utf8") as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            if limit is not None and i >= limit:
                return
            yield row[1], base / "audio" / split / row[1], row[3], "fleurs"


def recording_kit(root: Path) -> Iterator[tuple[str, Path, str, str]]:
    """Team recordings `<spk>_<NEU|URG>_<sid>.wav|flac`; reference from recording_kit/script_vi.csv; tag = NEU or URG."""
    script = {r["sentence_id"]: r["text_vi"] for r in csv.DictReader(open(ROOT / "recording_kit/script_vi.csv", encoding="utf-8-sig"))}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in {".wav", ".flac"}:
            continue
        spk, tag, sid = p.stem.split("_")
        yield p.stem, p, script[sid], tag
