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
    def __init__(self, model_dir: Path = DEFAULT_DIR, threads: int = 2, max_new_tokens: int = 48) -> None:
        d = Path(model_dir)
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
        self.enc = ort.InferenceSession(str(d / "encoder_model_quantized.onnx"), so, providers=p)
        self.dec = ort.InferenceSession(str(d / "decoder_model_merged_quantized.onnx"), so, providers=p)
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

    def translate(self, text: str, src: Lang, tgt: Lang) -> MtResult:
        assert (src, tgt) == ("vi", "en"), (src, tgt)
        out, _ = self.greedy(self.encode_ids(text))
        return MtResult(src_text=text, tgt_text=self.decode_ids(out), src_lang=src, tgt_lang=tgt, hops=["vi>en"])
