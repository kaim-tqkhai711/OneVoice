from types import SimpleNamespace
import numpy as np
import pytest
from tonebridge.stages.nmt_ort import OrtMarianNmt
from tonebridge.stages.nmt_marian import tokenizer_audit, ViEnNmt, EnViNmt, EnKoNmt, KoEnNmt


def toy(tokens, budget=8):
    model = OrtMarianNmt.__new__(OrtMarianNmt)
    model.pad, model.eos, model.max_new = 0, 1, budget
    model.enc = SimpleNamespace(run=lambda *args: [np.zeros((1, 1, 1), np.float32)])
    steps = iter(tokens)
    def run(*args):
        logits = np.full((1, 1, 8), -20., dtype=np.float32)
        logits[0, 0, next(steps)] = 20
        return [logits]
    model.dec = SimpleNamespace(run=run)
    model._empty_past = lambda n: {}
    model.dec_out = ["logits"]
    return model


def test_eos_is_real_not_inferred_from_budget():
    ids, info = toy([2, 1]).greedy([2])
    assert ids == [2] and info["eos_reached"] and not info["truncated"]
    assert info["decoder_steps"] == 2
    ids, info = toy([2, 3], budget=2).greedy([2])
    assert ids == [2, 3] and not info["eos_reached"] and info["truncated"]
    assert info["decoder_steps"] == 2


def test_partial_forced_phrase_is_not_satisfied():
    ids, info = toy([2], budget=1).greedy([2], [[[2, 3]]])
    assert ids == [2] and info["constraints_satisfied"] == 0
    assert info["unsatisfied_constraints"] == ["0"]


def test_duplicate_constraints_need_distinct_emitted_occurrences():
    _, info = toy([2, 1]).greedy([2], [[[2]], [[2]]])
    assert info["constraints_requested"] == 2 and info["constraints_satisfied"] == 1


def test_shared_prefix_does_not_report_all_alternatives_as_covered():
    ids, info = toy([2, 4, 1]).greedy([2], [[[2, 3], [2, 4]]])
    assert ids == [2, 3] and info["constraints_satisfied"] == 1


def test_constraints_do_not_silently_ignore_beam_config():
    model = toy([])
    model.source_limit = 20
    model.num_beams = 4
    model.encode_ids = lambda text: [2, 1]
    model.constrainer = lambda *args: []
    model.piece_ids = lambda text: []
    with pytest.raises(ValueError, match="constrained_beam"):
        model.translate("xin chào", "vi", "en")


def test_empty_overlong_and_wrong_direction_rejected_before_model():
    model = toy([])
    model.source_limit = 1
    model.encode_ids = lambda text: [2, 1]
    for text, src, tgt, message in [
        ("", "vi", "en", "empty_source"),
        ("hello", "en", "vi", "direction_mismatch"),
        ("xin chào", "vi", "en", "source_too_long")
    ]:
        with pytest.raises(ValueError, match=message):
            model.translate(text, src, tgt)


class FakeTokenizer:
    unk_token_id = 3
    target_encoder = {"x": 0, "</s>": 1, "<pad>": 2, "<unk>": 3}
    def get_src_vocab(self):
        return self.target_encoder
    def __call__(self, text, **kwargs):
        return {"input_ids": [0, 1]}


def test_tokenizer_audit_checks_mapping_and_unknowns():
    tok = FakeTokenizer()
    cfg = SimpleNamespace(vocab_size=4, decoder_vocab_size=4)
    assert tokenizer_audit(tok, cfg, ["Hello"]) == []
    tok.spm_source = SimpleNamespace(get_piece_size=lambda: 1, id_to_piece=lambda i: "missing")
    assert "source_sentencepiece_vocab_mismatch:1" in tokenizer_audit(tok, cfg, ["Hello"])


@pytest.mark.parametrize("adapter", [ViEnNmt, EnViNmt, EnKoNmt, KoEnNmt])
def test_each_adapter_missing_assets_is_explicit(tmp_path, adapter):
    with pytest.raises(FileNotFoundError, match="NMT assets missing"):
        adapter(tmp_path / "missing")
