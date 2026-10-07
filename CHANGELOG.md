# Changelog

This changelog contains the most important changes in this project. Versions follow semantic versioning: a patch release fixes code without changing any result, a minor release adds features, a major release changes the model or the way it runs.

## v2.0.0: FS²-SSA (2026-10-07)

The project is renamed **FS²-SSA** (`FS2-SSA` in file names and code). The square stands for the two FS of the design: **few-spike** coding and **softmax-free** attention.

### Added

* **Recurrent form of the attention.** `CausalSpikingSelfAttention` can now mix over time with a fixed state per head instead of the T x T matrix: `S_t = γ S_{t-1} + k_tᵀ v_t`, `Z_t = γ Z_{t-1} + 1`, `out_t = scale · (q_t S_t) / Z_t`. The cost per token no longer depends on how many tokens came before.

* `FSGPT.set_recurrent()`, `FSGPT.step()` and `FSGPT.generate_recurrent()` for inference one token at a time.

* **Verified on the published v1 checkpoint (perplexity 37.14)**, with no retraining: in float64 the parallel and the recurrent forms agree to a relative difference of `6e-16` on the logits, with the same argmax on 2 x 1024 tokens. In float32 a spike whose input sits within rounding distance of its threshold can fire in one form and not in the other: 95.2% of the tokens keep the same argmax, and the validation loss is the same (3.65808 against 3.65814).

* `recurrent_check.py`, the script behind those numbers. Self-contained: it carries the classes of the trained model and takes the weights from Hugging Face or from Drive.

### Changed

* The parallel forward is bit-identical to v1, so training is unchanged and v1 checkpoints load with `strict=True`.

* `generate.py` is self-contained too, with the weights from Hugging Face or from Drive.

* The README is rewritten for FS²-SSA and the v1 README moves to `docs/v1.md`, with three descriptive slips corrected and every number unchanged: the per-head decay ladder is fixed, not data-dependent; the loss gap in the takeaways is +0.0829, as in the results table; the channel gain α is a learnable multiplier initialised in (0.1, 1), not a bounded one.

### Fixed

* `SignedFSNeuron` names its gain `alpha1`, as in the checkpoints. In v1 `model.py` called it `alpha`, so the published weights did not load into it.

### Known limits

* Constant cost per token holds up to the trained context of 1024 tokens: the position embeddings are learned only up to there.

## v1.0.1 (2026-10-03)

Fixes reported in #1. Method, results and checkpoints are unchanged.

### Fixed

* `spike` was called by the FS neurons but not defined in `model.py`. The `TriangularSpike` surrogate used for every run is now included, so `model.py` runs on its own outside the original Colab session;

* Two rows of `CONFIGS` shared the same name, so the sweep would have skipped the per-head decay ladder. That row now has its own name;

* Comments brought in line with the code: the header, the `CONFIGS` legend, the `decay_stats` docstring and the `TARGET_TOKENS` comment.

## v1.0.0 (2026-09-26)

First release.

* A ~94M parameter autoregressive model on FineWeb-Edu with softmax-free causal attention: Q, K, V coded by few-spike neurons at K=2, rows normalised by a per-head decay ladder;

* Validation perplexity **37.44** against **34.46** for the dense softmax + GELU control at the same size, context and token budget (ΔLoss = +0.0829 nats);

* Ablation over five spiking variants, with the training histories in `results/history`;

* Evaluation scripts for diagnostics, figures and text generation.
