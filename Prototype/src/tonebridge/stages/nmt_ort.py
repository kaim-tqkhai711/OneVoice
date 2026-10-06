"""Real VI->EN NMT: opus-mt-vi-en (Marian) INT8 ONNX, run by ORT with a hand-written greedy loop.

Inference deps: onnxruntime + sentencepiece + numpy only (no torch / transformers), so the loop ports 1:1 to Kotlin
(port debt P1 tokenizer, P2 decode loop). Tokenizer = source.spm piece -> vocab.json id, exactly what MarianTokenizer does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from ..contracts import Lang, MtResult

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "models/nmt/vi-en-int8-arm64"


class OrtMarianNmt:
    def __init__(self, model_dir: Path = DEFAULT_DIR, threads: int = 2, max_new_tokens: int = 48,
                 enc_file: str = "encoder_model_quantized.onnx", dec_file: str = "decoder_model_merged_quantized.onnx",
                 num_beams: int = 1, length_penalty: float = 1.0) -> None:
        d = Path(model_dir)
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

    def greedy(self, src_ids: list[int]) -> tuple[list[int], dict[str, float]]:
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
        for step in range(self.max_new):
            feed = {"encoder_attention_mask": mask, "input_ids": np.asarray([[cur]], np.int64), "encoder_hidden_states": h,
                    "use_cache_branch": np.asarray([step > 0]), **past}
            res = self.dec.run(None, feed)
            logits = res[0][0, -1].copy()  # [V]
            logits[self.pad] = -np.inf  # bad_words_ids in generation_config
            cur = int(np.argmax(logits))
            if cur == self.eos:
                break
            out_ids.append(cur)
            pres = dict(zip(self.dec_out[1:], res[1:]))
            for n in past:  # past_key_values.L.{decoder,encoder}.{key,value} <- present.L....
                if step == 0 or ".decoder." in n:  # cached steps return empty cross-attn presents: keep step-0 encoder K/V
                    past[n] = pres[n.replace("past_key_values", "present")]
        return out_ids, {"encoder_ms": enc_ms, "decoder_steps": float(len(out_ids) + 1)}

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
            if len(finished) >= k and nxt and max(f[0] for f in finished) >= nxt[0][0] / ((step + 1) ** lp_pow if lp_pow > 0 else 1.0):
                break  # best finished beats the best running hypothesis' (optimistic) normalised score
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
        if not finished:
            finished = [(scores[i] / (max(len(beams[i]), 1) ** lp_pow), beams[i]) for i in range(k)]
        best = max(finished, key=lambda f: f[0])[1]
        return best, {"encoder_ms": enc_ms, "decoder_steps": float(steps)}

    def translate(self, text: str, src: Lang, tgt: Lang) -> MtResult:
        assert (src, tgt) == ("vi", "en"), (src, tgt)
        ids = self.encode_ids(text)
        out, _ = self.greedy(ids) if self.num_beams <= 1 else self.beam(ids, self.num_beams)
        return MtResult(src_text=text, tgt_text=self.decode_ids(out), src_lang=src, tgt_lang=tgt, hops=["vi>en"])
