#!/usr/bin/env bash
# Bake-off candidate: SMaLL-100 (MIT, 332.7M params) direct vi->ko. Timebox 2 h.
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python
D=models/nmt/small100-src
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("alirezamsh/small100", local_dir="models/nmt/small100-src",
                  allow_patterns=["config.json","model.safetensors","sentencepiece.bpe.model","special_tokens_map.json",
                                  "tokenization_small100.py","tokenizer_config.json","vocab.json","README.md"])
PYEOF
$PY -m optimum.commands.optimum_cli export onnx --model $D --task text2text-generation-with-past models/nmt/small100-fp32
for f in tokenization_small100.py sentencepiece.bpe.model vocab.json tokenizer_config.json special_tokens_map.json; do cp $D/$f models/nmt/small100-fp32/ ; done
$PY tools/bench_hop.py --model-dir models/nmt/small100-fp32 --tag vi-ko-small100 --hf-id alirezamsh/small100 \
   --threads 2 --sentences-file configs/sentences_vi.txt --tokenizer-kind small100 --tgt-lang ko
