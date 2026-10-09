"""Real HF CPU generation using tiny RANDOM test models; no pretrained download.

Validates local-only constructors, source languages and true termination evidence.
These tests say nothing about clinical translation accuracy.
"""
import json
import shutil
import pytest


@pytest.fixture(scope="module")
def tiny_assets(tmp_path_factory):
    import sentencepiece as spm
    from transformers import MarianConfig, MarianMTModel, MarianTokenizer
    d = tmp_path_factory.mktemp("tiny-random-marian")
    corpus = d / "corpus.txt"
    corpus.write_text("Hello. I need help. xin chào 안녕하세요.\n" * 20, encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(d / "sp"), vocab_size=48,
                                   hard_vocab_limit=False, unk_id=0, eos_id=1, pad_id=2, bos_id=-1,
                                   minloglevel=2, num_threads=1)
    sp = spm.SentencePieceProcessor(model_file=str(d / "sp.model"))
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    (d / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(d / "sp.model", d / "source.spm")
    shutil.copyfile(d / "sp.model", d / "target.spm")
    tokenizer = MarianTokenizer(source_spm=str(d / "source.spm"), target_spm=str(d / "target.spm"),
                                vocab=str(d / "vocab.json"))
    tokenizer.save_pretrained(d)
    cfg = MarianConfig(vocab_size=len(vocab), decoder_vocab_size=len(vocab), d_model=16,
                       encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                       decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                       max_position_embeddings=128, pad_token_id=2, eos_token_id=1,
                       decoder_start_token_id=2, forced_eos_token_id=1)
    model = MarianMTModel(cfg)
    # Force a non-EOS token at every step. Adapter must disable forced terminal
    # EOS rather than falsely treating a length-budget stop as complete.
    model.final_logits_bias.fill_(-100)
    model.final_logits_bias[0, 3] = 100
    model.save_pretrained(d, safe_serialization=True)
    return d


@pytest.mark.parametrize("src,tgt,text", [
    ("vi", "en", "xin chào"), ("en", "vi", "Hello."),
    ("en", "ko", "I need help."), ("ko", "en", "안녕하세요.")
])
def test_four_adapters_real_local_generation(tiny_assets, src, tgt, text):
    from tonebridge.stages.nmt_marian import MarianTextAdapter
    from tonebridge.contracts import MtResult
    model = MarianTextAdapter(tiny_assets, src, tgt, max_new_tokens=3)
    result = model.translate(text, src, tgt)
    assert isinstance(result, MtResult)
    assert (result.src_lang, result.tgt_lang) == (src, tgt)
    assert result.evidence.truncated and not result.evidence.eos_reached
    assert result.evidence.output_tokens == 3
    assert result.evidence.source_unknown_tokens == 0
