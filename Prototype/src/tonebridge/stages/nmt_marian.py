"""Four direct CPU Marian adapters. Local assets only; never download during inference."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import sys
from tonebridge.nmt_evidence import DIRECTIONS, EvidenceMtResult, TranslationEvidence


def tokenizer_audit(tokenizer, model_config, samples: list[str]) -> list[str]:
    """Validate token IDs against *source* embedding size and target vocabulary.

    Do not invent a mapping from SentencePiece IDs: embedding IDs must match
    the published checkpoint. Quarantine mismatched assets instead of silently
    converting unknown pieces to UNK.
    """
    issues = []
    src_vocab = tokenizer.get_src_vocab() if hasattr(tokenizer, "get_src_vocab") else tokenizer.get_vocab()
    limit = getattr(model_config, "vocab_size")
    if getattr(model_config, "separate_vocabs", False):
        limit = getattr(model_config, "encoder_vocab_size", limit)
    if max(src_vocab.values(), default=-1) >= limit:
        issues.append("source_ids_outside_embedding")
    target = getattr(tokenizer, "target_encoder", None) or tokenizer.get_vocab()
    if max(target.values(), default=-1) >= getattr(model_config, "decoder_vocab_size", model_config.vocab_size):
        issues.append("target_ids_outside_embedding")
    sp = getattr(tokenizer, "spm_source", None)
    if sp is not None:
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        missing = [piece for piece in pieces if piece not in src_vocab and not piece.startswith("<")]
        if missing:
            issues.append(f"source_sentencepiece_vocab_mismatch:{len(missing)}")
    for text in samples:
        encoded = tokenizer(text, truncation=False)["input_ids"]
        if tokenizer.unk_token_id in encoded:
            issues.append("probe_contains_source_unk:" + text)
    return sorted(set(issues))


class MarianTextAdapter:
    def __init__(self, model_dir: Path, src_lang: str, tgt_lang: str, threads: int = 2,
                 max_new_tokens: int = 128, num_beams: int = 1):
        direction = f"{src_lang}-{tgt_lang}"
        if direction not in DIRECTIONS:
            raise ValueError("unsupported_direction:" + direction)
        if threads < 1 or max_new_tokens < 1 or num_beams < 1:
            raise ValueError("positive threads/token budget/beam count required")
        d = Path(model_dir)
        if not d.is_dir():
            raise FileNotFoundError("NMT assets missing: " + str(d))
        if sys.platform == "win32" and not sys.flags.utf8_mode:
            raise ValueError("Windows Marian tokenizer requires Python -X utf8 (or PYTHONUTF8=1)")
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        torch.set_num_threads(threads)
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.max_new_tokens, self.num_beams = max_new_tokens, num_beams
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        separate = bool(cfg.get("separate_vocabs", False))
        if separate and not (d / "target_vocab.json").is_file():
            raise ValueError("separate_vocabulary_requires_target_vocab.json")
        manifest = d / "asset_manifest.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        self.model_id = metadata.get("model_id", str(d))
        self.revision = metadata.get("revision", "")
        self.provenance_issues = []
        if not metadata.get("files") or not re.fullmatch("[0-9a-f]{40}", self.revision):
            self.provenance_issues.append("asset_provenance_unverified")
        else:
            files = metadata["files"]
            required = {"config.json", "source.spm", "target.spm", "vocab.json"}
            if not required.issubset(files) or not any(name.endswith(".safetensors") for name in files):
                raise ValueError("incomplete_asset_manifest")
            actual = {path.name for path in d.iterdir() if path.is_file()
                      and path.suffix in (".json", ".spm", ".safetensors") and path.name != "asset_manifest.json"}
            if not actual.issubset(files):
                raise ValueError("unlisted_asset_files")
            for name, expected in files.items():
                path = (d / name).resolve()
                if not path.is_relative_to(d.resolve()):
                    raise ValueError("asset_manifest_path_escape")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                    raise ValueError("asset_integrity_mismatch:" + name)
        self.tokenizer = MarianTokenizer.from_pretrained(
            str(d), local_files_only=True, separate_vocabs=separate)
        self.model = MarianMTModel.from_pretrained(
            str(d), local_files_only=True, use_safetensors=True).eval().cpu()
        self.source_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        self.tokenizer_issues = tokenizer_audit(
            self.tokenizer, self.model.config,
            {"vi": ["xin chào"], "en": ["Hello.", "I need help."], "ko": ["안녕하세요."]}[src_lang])
        if self.tokenizer_issues:
            raise ValueError("tokenizer_validation_failed:" + ";".join(self.tokenizer_issues))

    def translate(self, text: str, src: str, tgt: str) -> EvidenceMtResult:
        if (src, tgt) != (self.src_lang, self.tgt_lang):
            raise ValueError(f"direction_mismatch:{src}-{tgt}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty_source")
        if len(text) > 8192:
            raise ValueError("source_character_limit_exceeded")
        import torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=False)
        source = enc["input_ids"][0].tolist()
        if len(source) > self.source_limit:
            raise ValueError(f"source_too_long:{len(source)}>{self.source_limit}")
        with torch.inference_mode():
            output = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                         num_beams=self.num_beams, do_sample=False,
                                         forced_eos_token_id=None)[0].tolist()
        generated = output[1:]  # decoder_start token is not an output token
        eos = self.model.config.eos_token_id
        reached = eos in generated
        if reached:
            generated = generated[:generated.index(eos)]
        decoded = self.tokenizer.decode(output, skip_special_tokens=True)
        problems = list(self.tokenizer_issues) + self.provenance_issues
        if "\ufffd" in decoded:
            problems.append("replacement_character")
        if tgt == "ko" and not re.search("[가-힣]", decoded):
            problems.append("expected_hangul_missing")
        if re.search("[\u0400-\u04ff]", decoded) or (tgt in ("vi", "en") and re.search("[가-힣]", decoded)):
            problems.append("unexpected_output_script")
        evidence = TranslationEvidence(
            eos_reached=reached, truncated=not reached, source_tokens=len(source),
            output_tokens=len(generated), source_unknown_tokens=source.count(self.tokenizer.unk_token_id),
            output_unknown_tokens=generated.count(self.tokenizer.unk_token_id),
            tokenizer_issues=problems, model_id=self.model_id, revision=self.revision, backend="torch-cpu")
        return EvidenceMtResult(src_text=text, tgt_text=decoded, src_lang=src, tgt_lang=tgt,
                                hops=[f"{src}>{tgt}"], evidence=evidence)


class ViEnNmt(MarianTextAdapter):
    def __init__(self, model_dir: Path, **kwargs):
        super().__init__(model_dir, "vi", "en", **kwargs)


class EnViNmt(MarianTextAdapter):
    def __init__(self, model_dir: Path, **kwargs):
        super().__init__(model_dir, "en", "vi", **kwargs)


class EnKoNmt(MarianTextAdapter):
    def __init__(self, model_dir: Path, **kwargs):
        super().__init__(model_dir, "en", "ko", **kwargs)


class KoEnNmt(MarianTextAdapter):
    def __init__(self, model_dir: Path, **kwargs):
        super().__init__(model_dir, "ko", "en", **kwargs)
