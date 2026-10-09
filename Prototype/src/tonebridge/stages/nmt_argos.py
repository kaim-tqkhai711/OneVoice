"""Local EN->KO Argos CPU candidate; independent of quarantined Marian assets."""
import hashlib
import json
from pathlib import Path
import re

from tonebridge.nmt_evidence import EvidenceMtResult, TranslationEvidence
from tonebridge.nmt_source import prepare_source


class ArgosEnKoNmt:
    def __init__(self, model_dir: Path, threads=2, max_new_tokens=128, format_asr_source=False):
        if threads < 1 or max_new_tokens < 1:
            raise ValueError("positive threads and token budget required")
        d = Path(model_dir).resolve()
        metadata = json.loads((d / "asset_manifest.json").read_text(encoding="utf-8"))
        required = {"sentencepiece.model", "model/model.bin", "model/shared_vocabulary.txt", "metadata.json"}
        if not required.issubset(metadata["files"]) or not re.fullmatch("[0-9a-f]{40}", metadata["revision"]):
            raise ValueError("incomplete_asset_manifest")
        for name, item in metadata["files"].items():
            path = (d / name).resolve()
            if not path.is_relative_to(d):
                raise ValueError("asset_manifest_path_escape")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("asset_integrity_mismatch:" + name)
        spec = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        if (spec["from_code"], spec["to_code"]) != ("en", "ko"):
            raise ValueError("direction_mismatch")
        import sentencepiece as spm
        import ctranslate2
        self.sp = spm.SentencePieceProcessor(model_file=str(d / "sentencepiece.model"))
        self.vocab = set((d / "model/shared_vocabulary.txt").read_text(encoding="utf-8").splitlines())
        if not {"<unk>", "</s>"}.issubset(self.vocab):
            raise ValueError("missing_special_tokens")
        self.model = ctranslate2.Translator(str(d / "model"), device="cpu", compute_type="int8",
                                           inter_threads=1, intra_threads=threads)
        self.max_new_tokens = max_new_tokens
        self.format_asr_source = format_asr_source
        self.model_id, self.revision = metadata["model_id"], metadata["revision"]
        for probe in ("Hello.", "I need help."):
            if self.unknown_count(probe):
                raise ValueError("tokenizer_probe_unknown")

    def unknown_count(self, text):
        pieces = self.sp.encode(text, out_type=str)
        ids = self.sp.encode(text, out_type=int)
        return sum(i == self.sp.unk_id() or p not in self.vocab for i, p in zip(ids, pieces))

    def translate(self, text, src, tgt):
        if (src, tgt) != ("en", "ko"):
            raise ValueError("direction_mismatch")
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            raise ValueError("invalid_source")
        prepared, preparation = prepare_source(text, src, self.format_asr_source)
        source = self.sp.encode(prepared, out_type=str)
        if len(source) > 512:
            raise ValueError("source_too_long")
        result = self.model.translate_batch([source], beam_size=1, max_input_length=0,
                    max_decoding_length=self.max_new_tokens, return_end_token=True,
                    replace_unknowns=False, disable_unk=False)[0]
        generated = result.hypotheses[0]
        reached = bool(generated) and generated[-1] == "</s>"
        output = generated[:-1] if reached else generated
        decoded = self.sp.decode(output)
        issues = []
        if not re.search("[가-힣]", decoded):
            issues.append("expected_hangul_missing")
        if "\ufffd" in decoded:
            issues.append("replacement_character")
        ev = TranslationEvidence(eos_reached=reached, truncated=not reached,
              source_tokens=len(source), output_tokens=len(output),
              source_unknown_tokens=self.unknown_count(prepared), output_unknown_tokens=output.count("<unk>"),
              tokenizer_issues=issues, model_id=self.model_id, revision=self.revision, backend="ctranslate2-int8-cpu",
              source_preparation=preparation)
        return EvidenceMtResult(src_text=text, tgt_text=decoded, src_lang=src, tgt_lang=tgt,
                                hops=["en>ko"], evidence=ev)
