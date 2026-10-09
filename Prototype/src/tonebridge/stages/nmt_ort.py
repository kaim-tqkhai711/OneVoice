"""Real VI->EN NMT: opus-mt-vi-en (Marian) INT8 ONNX, run by ORT with a hand-written greedy loop.

Inference deps: onnxruntime + sentencepiece + numpy only (no torch / transformers), so the loop ports 1:1 to Kotlin
(port debt P1 tokenizer, P2 decode loop). Tokenizer = source.spm piece -> vocab.json id, exactly what MarianTokenizer does.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from ..contracts import Lang, MtResult
from ..nmt_evidence import EvidenceMtResult, TranslationEvidence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "models/nmt/vi-en-int8-arm64"


class OrtMarianNmt:
    def __init__(self, model_dir: Path = DEFAULT_DIR, threads: int = 2, max_new_tokens: int = 48,
                 enc_file: str = "encoder_model_quantized.onnx", dec_file: str = "decoder_model_merged_quantized.onnx",
                 num_beams: int = 1, length_penalty: float = 1.0) -> None:
        d = Path(model_dir)
        if max_new_tokens < 1 or threads < 1 or num_beams < 1:
            raise ValueError("positive token budget, threads and beam count required")
        self.num_beams, self.length_penalty = num_beams, length_penalty
        cfg = json.loads((d / "config.json").read_text(encoding="utf8"))
        self.eos, self.pad = cfg["eos_token_id"], cfg["pad_token_id"]  # decoder start = pad
        self.n_layers, self.heads = cfg["decoder_layers"], cfg["decoder_attention_heads"]
        self.head_dim = cfg["d_model"] // self.heads
        self.max_new = max_new_tokens
        self.vocab: dict[str, int] = json.loads((d / "vocab.json").read_text(encoding="utf8"))
        self.inv = {v: k for k, v in self.vocab.items()}
        self.unk = self.vocab["<unk>"]
        self.sp_src = spm.SentencePieceProcessor(model_file=str(d / "source.spm"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=str(d / "target.spm"))
        self.source_limit = int(cfg.get("max_position_embeddings", 512))
        self.model_dir = d
        self.model_id, self.revision = str(d), ""
        self.provenance_issues = []
        manifest = d / "asset_manifest.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            required = {"config.json", "source.spm", "target.spm", "vocab.json", enc_file, dec_file}
            if not required.issubset(metadata.get("files", {})) or not re.fullmatch("[0-9a-f]{40}", metadata.get("revision", "")):
                raise ValueError("incomplete_asset_manifest")
            for name, expected in metadata["files"].items():
                path = (d / name).resolve()
                if not path.is_relative_to(d.resolve()):
                    raise ValueError("asset_manifest_path_escape")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                    raise ValueError("asset_integrity_mismatch:" + name)
            self.model_id, self.revision = metadata["model_id"], metadata["revision"]
        else:
            self.provenance_issues = ["asset_provenance_unverified"]
        self.vocabulary_warnings = []
        for sp in (self.sp_src, self.sp_tgt):
            missing = [sp.id_to_piece(i) for i in range(sp.get_piece_size())
                       if sp.id_to_piece(i) not in self.vocab and not sp.id_to_piece(i).startswith("<")]
            if missing:
                # Preserve the published pruned vocabulary. Never invent IDs;
                # unknown source pieces are counted and the runtime blocks them.
                self.vocabulary_warnings.append(f"sentencepiece_vocab_mismatch:{len(missing)}")
        if min(self.vocab.values()) < 0 or max(self.vocab.values()) >= cfg["vocab_size"] or len(self.inv) != len(self.vocab):
            raise ValueError("invalid_published_vocabulary_ids")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        p = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / enc_file), so, providers=p)
        self.dec = ort.InferenceSession(str(d / dec_file), so, providers=p)
        self.dec_in = [i.name for i in self.dec.get_inputs()]
        self.dec_out = [o.name for o in self.dec.get_outputs()]  # logits, present.*

    def encode_ids(self, text: str) -> list[int]:
        return [self.vocab.get(p, self.unk) for p in self.sp_src.encode(text, out_type=str)] + [self.eos]

    def decode_ids(self, ids: list[int]) -> str:
        return self.sp_tgt.decode_pieces([self.inv[i] for i in ids if i not in (self.eos, self.pad)])

    def _empty_past(self, enc_len: int) -> dict[str, np.ndarray]:
        """Step-0 dummies for the merged decoder: self-attn past has length 0, cross-attn past has encoder length."""
        z = {"decoder": np.zeros((1, self.heads, 0, self.head_dim), np.float32),
             "encoder": np.zeros((1, self.heads, enc_len, self.head_dim), np.float32)}
        return {n: z[n.split(".")[2]] for n in self.dec_in if n.startswith("past_key_values")}

    def greedy(self, src_ids: list[int], constraints: list[list[list[int]]] | None = None, bonus: float = 0.0,
               eos_penalty: float | None = None) -> tuple[list[int], dict]:
        """Returns generated ids (no start/eos) and {"encoder_ms", "decoder_steps"} for composed-latency accounting."""
        import time

        ids = np.asarray([src_ids], np.int64)  # [1, S]
        mask = np.ones_like(ids)
        t = time.perf_counter()
        h = self.enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]  # [1, S, 512]
        enc_ms = (time.perf_counter() - t) * 1000
        past = self._empty_past(ids.shape[1])
        out_ids: list[int] = []
        cur = self.pad
        # Soft lexical constraints (glossary): each constraint = list of accepted token-id sequences. Unsatisfied constraints add `bonus` to the first
        # token of every alternative (and subtract `eos_penalty` from EOS); once a first token is emitted the rest of that alternative is forced.
        cons = constraints or []
        sat = [False] * len(cons)
        active: tuple[int, list[int], int] | None = None
        eos_pen = bonus if eos_penalty is None else eos_penalty
        steps, eos_reached = 0, False
        for step in range(self.max_new):
            feed = {"encoder_attention_mask": mask, "input_ids": np.asarray([[cur]], np.int64), "encoder_hidden_states": h,
                    "use_cache_branch": np.asarray([step > 0]), **past}
            res = self.dec.run(None, feed)
            steps += 1
            logits = res[0][0, -1].copy()  # [V]
            logits[self.pad] = -np.inf  # bad_words_ids in generation_config
            if active is not None:
                ci, alt, pos = active
                cur = alt[pos]
                if pos + 1 >= len(alt):
                    sat[ci], active = True, None
                else:
                    active = (ci, alt, pos + 1)
            else:
                pend = [i for i in range(len(cons)) if not sat[i]]
                if pend and bonus:
                    for i in pend:
                        for alt in cons[i]:
                            logits[alt[0]] += bonus
                    logits[self.eos] -= eos_pen
                cur = int(np.argmax(logits))
                for i in pend:
                    for alt in cons[i]:
                        if cur == alt[0]:
                            if len(alt) == 1:
                                sat[i] = True
                            else:
                                active = (i, alt, 1)
                            break
                    if sat[i] or active is not None:
                        break
            if cur == self.eos:
                eos_reached = True
                break
            out_ids.append(cur)
            pres = dict(zip(self.dec_out[1:], res[1:]))
            for n in past:  # past_key_values.L.{decoder,encoder}.{key,value} <- present.L....
                if step == 0 or ".decoder." in n:  # cached steps return empty cross-attn presents: keep step-0 encoder K/V
                    past[n] = pres[n.replace("past_key_values", "present")]
        # Verify complete emitted phrases; never report a first-token hit as coverage.
        covered = []
        occupied = set()
        for ci, alternatives in enumerate(cons):
            for alt in alternatives:
                start = next((j for j in range(len(out_ids) - len(alt) + 1)
                              if out_ids[j:j + len(alt)] == alt
                              and not occupied.intersection(range(j, j + len(alt)))), None)
                if start is not None:
                    occupied.update(range(start, start + len(alt)))
                    covered.append(ci)
                    break
        return out_ids, {"encoder_ms": enc_ms, "decoder_steps": steps,
                         "eos_reached": eos_reached, "truncated": not eos_reached,
                         "constraints_requested": len(cons), "constraints_satisfied": len(covered),
                         "unsatisfied_constraints": [str(i) for i in range(len(cons)) if i not in covered]}

    def beam(self, src_ids: list[int], k: int = 4, length_penalty: float | None = None,
             bias: dict[int, float] | None = None) -> tuple[list[int], dict[str, float]]:
        """Beam search over the merged decoder with batch = k. Hypothesis score = sum(log p) / len**length_penalty (len counts the EOS).
        ``bias``: optional {token_id: additive log-prob bonus} applied every step (used for glossary-constrained decoding).
        All beams share the same encoder states, so cross-attention K/V never need re-ordering."""
        import time

        lp_pow = self.length_penalty if length_penalty is None else length_penalty
        ids = np.asarray([src_ids], np.int64)
        mask1 = np.ones_like(ids)
        t = time.perf_counter()
        h1 = self.enc.run(None, {"input_ids": ids, "attention_mask": mask1})[0]  # [1, S, 512]
        enc_ms = (time.perf_counter() - t) * 1000
        h, mask = np.repeat(h1, k, axis=0), np.repeat(mask1, k, axis=0)  # [k, S, 512], [k, S]
        past = {n: np.repeat(v, k, axis=0) for n, v in self._empty_past(ids.shape[1]).items()}
        beams: list[list[int]] = [[] for _ in range(k)]
        scores = np.array([0.0] + [-1e9] * (k - 1))
        cur = np.full((k, 1), self.pad, np.int64)
        finished: list[tuple[float, list[int]]] = []
        steps = 0
        for step in range(self.max_new):
            feed = {"encoder_attention_mask": mask, "input_ids": cur, "encoder_hidden_states": h,
                    "use_cache_branch": np.asarray([step > 0]), **past}
            res = self.dec.run(None, feed)
            steps += 1
            logits = res[0][:, -1, :].astype(np.float64)  # [k, V]
            logits[:, self.pad] = -np.inf
            logp = logits - (np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) + logits.max(1, keepdims=True))
            if bias:
                for tid, b in bias.items():
                    logp[:, tid] += b
            cand = (scores[:, None] + logp).reshape(-1)
            top = np.argsort(-cand)[: 2 * k]
            nxt: list[tuple[float, int, int]] = []
            for c in top:
                bi, tok = divmod(int(c), logp.shape[1])
                sc = float(cand[c])
                if tok == self.eos:
                    if sc > -1e8:
                        finished.append((sc / ((len(beams[bi]) + 1) ** lp_pow), beams[bi]))
                else:
                    nxt.append((sc, bi, tok))
                if len(nxt) == k:
                    break
            if (len(finished) >= k and nxt and not bias and lp_pow >= 0
                    and max(f[0] for f in finished) >= max(sc for sc, _, _ in nxt) / ((self.max_new + 1) ** lp_pow)):
                break  # upper bound uses maximum future length for negative log scores
            if not nxt:
                break
            sel = np.array([bi for _, bi, _ in nxt] + [nxt[-1][1]] * (k - len(nxt)))
            beams = [beams[bi] + [tok] for _, bi, tok in nxt] + [beams[nxt[-1][1]] + [nxt[-1][2]]] * (k - len(nxt))
            scores = np.array([sc for sc, _, _ in nxt] + [-1e9] * (k - len(nxt)))
            cur = np.array([[tok] for _, _, tok in nxt] + [[nxt[-1][2]]] * (k - len(nxt)), np.int64)
            pres = dict(zip(self.dec_out[1:], res[1:]))
            for n in past:
                if step == 0 or ".decoder." in n:
                    v = pres[n.replace("past_key_values", "present")]
                    past[n] = v[sel] if ".decoder." in n else v
        eos_reached = bool(finished)
        if not finished:
            finished = [(scores[i] / (max(len(beams[i]), 1) ** lp_pow), beams[i]) for i in range(k)]
        best = max(finished, key=lambda f: f[0])[1]
        return best, {"encoder_ms": enc_ms, "decoder_steps": steps,
                      "eos_reached": eos_reached, "truncated": not eos_reached}

    def piece_ids(self, word: str) -> list[int]:
        """Target-side token ids for a surface form (SentencePiece pieces -> vocab ids)."""
        return [self.vocab.get(p, self.unk) for p in self.sp_tgt.encode(word, out_type=str)]

    constrainer = None  # optional callable (src_text, piece_ids) -> constraints; set by the factory when the glossary lever is on
    constraint_bonus = 0.0

    def translate(self, text: str, src: Lang, tgt: Lang) -> MtResult:
        if (src, tgt) != ("vi", "en"):
            raise ValueError(f"direction_mismatch:{src}-{tgt}")
        if not text.strip():
            raise ValueError("empty_source")
        ids = self.encode_ids(text)
        if len(ids) > self.source_limit:
            raise ValueError(f"source_too_long:{len(ids)}>{self.source_limit}")
        if getattr(self, "constrainer", None) is not None:
            if self.num_beams > 1:
                raise ValueError("constrained_beam_not_implemented; use greedy explicitly")
            out, info = self.greedy(ids, self.constrainer(text, self.piece_ids), self.constraint_bonus)
        else:
            out, info = self.greedy(ids) if self.num_beams <= 1 else self.beam(ids, self.num_beams)
        evidence = TranslationEvidence(
            eos_reached=info["eos_reached"], truncated=info["truncated"],
            source_tokens=len(ids), output_tokens=len(out),
            source_unknown_tokens=ids.count(self.unk), output_unknown_tokens=out.count(self.unk),
            constraints_requested=info.get("constraints_requested", 0),
            constraints_satisfied=info.get("constraints_satisfied", 0),
            unsatisfied_constraints=info.get("unsatisfied_constraints", []),
            tokenizer_issues=getattr(self, "provenance_issues", []),
            model_id=getattr(self, "model_id", str(self.model_dir)), revision=getattr(self, "revision", ""), backend="onnx-cpu")
        return EvidenceMtResult(src_text=text, tgt_text=self.decode_ids(out), src_lang=src,
                                tgt_lang=tgt, hops=["vi>en"], evidence=evidence)
