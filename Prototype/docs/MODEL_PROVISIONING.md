# Provisioning pinned laptop models

Run from `Prototype/` in a virtualenv installed from `requirements-bench.txt`.
Inference uses `requirements-runtime.txt`; export tooling is not imported by
the runtime. `models/` stays ignored by Git.

**Verified 15 October 2026:** the original merged decoder is installed and
matches the existing pin. Both preflights exit 0 (**368/368** bench,
**370/370** full). Both requested real Branch A benchmarks completed
**100/100 audio turns**; the regression suite is **466 passed, 9 skipped**.
See [Milestone C, section 10](../reports/MILESTONE_C_2026-10-14.md#10-nghiệm-thu-branch-a-model-thật--15102026)
for measured cold/warm latency, process-tree RSS and limitations. No manifest
hash was changed. The older blocker description below is historical.

```powershell
.venv/clean/Scripts/python.exe tools/provision_models.py --mode branch-a-bench --out results/provision_branch_a.json
.venv/clean/Scripts/python.exe tools/provision_models.py --mode full --out results/provision_full.json
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode branch-a-bench
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode full
```

`branch-a-bench` selects 368 pinned files, `full` selects 370 with the default
OFF denoiser and unconfigured urgency classifier. Configured additional packs
need manifest entries and an original offline directory; the tool does not
guess a URL for unknown weights. Both tools read the same `pipeline.json` and
`runtime_files.json` as the factory.

| Pack | Official source / reconstruction | Terms |
|---|---|---|
| Silero | [sherpa release](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx) | Silero MIT |
| VI ASR encoder/joiner/tokens | [sherpa archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2) | zzasdf Apache-2.0 model card |
| VI ASR INT8 decoder | [zzasdf immutable revision](https://huggingface.co/zzasdf/viet_iter3_pseudo_label/tree/e827965a37aab92a4455566fac49c0e80a23afef/exp) | Apache-2.0 |
| Marian VI→EN | [Helsinki immutable revision](https://huggingface.co/Helsinki-NLP/opus-mt-vi-en/tree/c8d2853e77f5fae31124d993e0b35176b1c8914e); FP32 Optimum export, dynamic INT8 arm64, per_channel=False | Apache-2.0; Optimum Apache-2.0 |
| Piper + espeak data | [sherpa archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ljspeech-medium.tar.bz2), [voice origin](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ljspeech/medium), [pack model card](https://huggingface.co/csukuangfj/vits-piper-en_US-ljspeech-medium/blob/34f390d13b9edbaa30c339127de0f9c4c5948c90/MODEL_CARD) | LJ Speech public domain data; bundled espeak-ng GPL-3.0; existing distribution/linking review remains in TTS_GPL_OPTIONS.md |
| SwiftF0 | Installed [0.3.0 wheel](https://pypi.org/project/swift-f0/0.3.0/), or official PyPI wheel download | MIT |
| Band energies | Local `tonebridge.branch_b_graph.build`, ONNX 1.18.0 + NumPy 2.3.5 | Project graph |

GitHub archive URLs are release assets; every installed member is authenticated
against the manifest, even if a release asset is replaced upstream. Hugging Face
checkpoint URLs use immutable revisions. Raw original Marian config/vocab are
not the exported ones; the tokenizer is saved with Transformers before checking
the runtime pins. The exporter never executes Hugging Face repository code.

Downloads are HTTPS, bounded and streamed to temporary directories under the
project. Tar extraction uses a whitelist and rejects traversal, links, duplicate
members and special files. Size and SHA-256 are both mandatory. A rejected file
is deleted, all temporary archives/source weights are cleaned, and exit status
is 1. Valid files are retained across retries; no valid target is overwritten.
An existing corrupted selected asset is deleted and the run stops immediately.
The lock prevents simultaneous provisioners; a stale lock requires an explicit
operator check. There is no hash override or manifest rewrite flag.

Installation checks a complete group before exposing its files; installation
errors roll back newly installed members. NMT tokenizer/config/encoder are an
independent verified subgroup. A missing decoder still prevents both preflights
and the factory from opening any inference session. No stub fallback is used.

## Historical merged-decoder blocker (14 October 2026; resolved 15 October)

The public source revision and quantization recipe reproduce **369/370 full
assets** (**367/368 branch-a-bench**). The reconstructed merged decoder has the
correct size, 81,649,685 bytes, but fails the original SHA-256:

```text
expected: ea35145e4b0b6757a9b1bde31c0363423115418fad23c0379cd1d3a9a2d5518d
actual:   14afbaeb4e4771d7866fd7286bdc759314fa045858027ee13a8b0416f65d3400
```

Seed 0 rebuilds using both ORT 1.22.1 and isolated ORT 1.23.2 produced that same
rejected digest. This is not an accepted replacement. The original merged
graph is not published in the Helsinki repository. The installed Optimum
`_create_name_sharing_dict()` chooses initializer names with `set.pop()`; these
names can depend on Python hash randomization. Freezing hash seed 0 makes the
current recipe repeatable, but does not establish the missing original export
environment or prove the sole cause of the mismatch. The historical report
mentions Python 3.10; the original hash seed is not recorded.

An original binary matching the expected hash is sufficient to unblock without
altering any runtime setting. Place it in a source directory at
`models/nmt/vi-en-int8-arm64/decoder_model_merged_quantized.onnx` and pass that
**models/** directory to the importer:

```powershell
.venv/clean/Scripts/python.exe tools/provision_models.py --mode full --from-directory D:/original-pack/models --out results/provision_original.json
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode branch-a-bench
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode full
```

Already verified targets are reused, so that source directory can contain just
the missing pinned decoder at the correct relative path. A different decoder
hash is rejected even if it runs or translates similarly. Do not change the
manifest to make an unverified candidate pass preflight.

`--nmt-quantizer-path` supports an explicitly supplied isolated original ORT
tooling directory during reconstruction experiments; it never changes the
runtime ORT dependency. It cannot substitute for the missing original graph.

## Benchmark inputs and verified execution

Three original ASR test WAVs are saved in `results/inputs/`; their immutable
URLs, SHA-256, mono/16-kHz validation and durations (3.74/2.40/1.55 seconds) are
in `provenance.json`. They are public speech samples, not synthetic stub tones
or a clinical accuracy corpus. `configs/pipeline_bench_whole.json` differs from
the runtime config only in `decode.tts_first_clause=false`.

The following two commands completed successfully on 15 October 2026. Their
result paths now exist: for a new run, choose fresh paths because the tool
refuses to overwrite evidence. Older `bench_real_branch_a.json` and
`bench_real_branch_a_whole.json` remain blocked historical diagnostics.

```powershell
.venv/clean/Scripts/python.exe tools/bench_e2e.py --mode branch-a-bench --wav results/inputs/zipformer_vi_0.wav results/inputs/zipformer_vi_1.wav results/inputs/zipformer_vi_2.wav --turns 100 --out results/bench_real_branch_a_verified.json
.venv/clean/Scripts/python.exe tools/bench_e2e.py --mode branch-a-bench --config configs/pipeline_bench_whole.json --wav results/inputs/zipformer_vi_0.wav results/inputs/zipformer_vi_1.wav results/inputs/zipformer_vi_2.wav --turns 100 --out results/bench_real_branch_a_whole_verified.json
```

Both configurations must synthesize every requested turn to reach COMPLETE.
Report the actual split coverage; an input translating to a single clause
cannot demonstrate a first-clause speedup. First buffer is readiness, not
physical audible onset. Repeat these public samples for latency/coverage only;
100 repeated turns are not 100 independent clinical examples. Full mode retains
uncalibrated quality checks and UNCONFIGURED urgency, without forced SPEAK.

The verified pair observed warm first-buffer p50/p95 **234.44/389.21 ms** versus
**240.16/413.80 ms**, and resident process-tree peaks **907.01/868.43 MiB**.
Separate metadata-only diagnostics found **0/3 multi-clause inputs**; the
observed latency differences do not establish a clause-streaming speedup.
Artifacts: [first-clause](../results/bench_real_branch_a_verified.json),
[whole-utterance](../results/bench_real_branch_a_whole_verified.json),
[comparison/coverage](../results/bench_real_branch_a_comparison_verified.json).
