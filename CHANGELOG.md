# Changelog

This changelog contains the most important changes in this project. Versions follow semantic versioning: a patch release fixes code without changing any result, a minor release adds features, a major release changes the model or the way it runs.

## Next

* Spiking output head, the last dense layer, coded into spikes from the first training step.

* The v2.0 hybrid trained with the scheduled surrogate, to measure what spiking the inputs still costs with the same backward pass.

* The two stages of the surrogate gain, apart: from one checkpoint in the stable phase of a warmup, stable, decay schedule, two cooldowns, one narrowing the surrogate and one keeping it fixed.

* Decay as a shift: values of `γ` rounded to `1 - 2^-n`, so that the decay of the state becomes a shift. A model with this decay needs a new training run.

* Spiking RMSNorm in the model: the reciprocal square root as a product of factors `1 + 2^-k`, each chosen by a K=2 FS neuron, so far tested as a standalone module.

## v2.2.0: scheduled surrogate (2026-11-01)

The K=3 model of v2.1 is trained again with a surrogate gradient sized to the threshold of each neuron and narrowed by a scheduler. The forward pass and the number of parameters do not change; the scripts, the results and the published weights of v2.0 and v2.1 are unchanged.

### Added

* **`v2.2.0/model.py`, the scheduled surrogate.** `TriangularSpike` takes its half-width as a tensor, one value per channel, with peak `1 / width` and so area 1, as the step it stands for: narrowing it changes the shape of the gradient, not its scale. In `FSNeuron` and `LearnableFSNeuron` the half-width is `width · _WIDTH_SCALE · s`, with `s` the top threshold of each channel, detached in the learnable neuron so that no gradient flows through the width. `width_scale_at` and `set_width_scale` keep it at `WIDTH = 1.5` through the warmup and bring it down to `WIDTH_END = 1.0` along the cosine of the learning rate: below `1.0 s` the backward pass of a code would exceed the slope of its forward pass. The evaluation line prints the gradient norm of the last step, before clipping. The forward pass is the one of v2.1.0, so a v2.2 checkpoint loads into the classes of v2.1 and a v2.1 checkpoint into those of v2.2.

* **One run**, `ssa K=3 +/- L g var alpha app data decay width sched`, the one entry of `CONFIGS`, from the same initial weights and on the same batches as the K=3 run of v2.1.0: validation loss **3.6145**, perplexity **37.13**, 0.0427 nats below v2.1.0 and level with the v2.0 hybrid (3.6228, and 3.6147 in its rerun). At equal steps the gap is 0.022 nats up to step 6,000 and 0.039 nats over steps 8,000 to 10,000. History in `v2.2.0/results/history`, checkpoint on Hugging Face as `v2.2.0/model.pt`.

* `v2.2.0/diagnostic.py`: the best of the four runs (dense control, v2.0 hybrid, v2.1 at K=3, v2.2) against the dense control, and the paired table of the two K=3 runs, with the half-width at each step. It finds every history by the name it stores, in `v2.0.0/`, `v2.1.0/` and `v2.2.0/`, and writes three figures to `v2.2.0/figures`, one point per evaluation: the validation loss of the four runs, their last six evaluations, and the paired difference in perplexity above the half-width.

* `v2.2.0/fluency.py`: 30 prompts (the three of `generate.py`, twelve written by hand, fifteen beginnings of validation documents with their human continuation), five continuations of 250 tokens per prompt and model with the same random draws, scored with GPT-2 large by rep-4, reference loss and PMI with the prompt, paired by prompt with 95% intervals from the t distribution. Each model takes its weights from Hugging Face or from Drive (`"hf"` or `"drive"` in `CONFIGS`), with progress bars. It writes every continuation with its scores, and a blind copy with the two models under letters. Between v2.2 and v2.1 no difference is distinguishable from zero.

* `v2.2.0/generate.py` and `v2.2.0/recurrent_check.py`, self-contained like those of v2.1: they carry the classes of `v2.2.0/model.py` and take the checkpoint from Hugging Face (`HF_FILE`, `v2.2.0/model.pt`) or from Drive.

* **Log-prob margins** in the README: agreement and plural bias on two minimal pairs, and the name, position and word effects on the four orderings of the binding prompt, for the v2.0 hybrid, v2.1 and v2.2.

### Changed

* `v2.1.0/diagnostic.py` tells the v2.2.0 run apart by its exact name (`V22_RUNS`); the name column of its table widens to 52 characters to fit it.

* The README is rewritten for v2.2: results, the section What is new in v2.2, the probes of v2.2 with the verdicts of v2.1 and v1 next to them, quick start, roadmap and repository tree. Its figures are those of `v2.2.0/figures`. The README of v2.1 moves to `docs/v2.1.md`, unchanged apart from its relative links.

* The reading of the two probes that v2.1 failed is corrected: the margins show that its `were` came from a bias towards the plural in that construction, and its `glaciers` from a preference for the word, not from the most recent candidate.

### Known limits

* The output head reads floats: 576 x 50,257 MACs per token.

* One seed per run. The gain mixes the shape of the surrogate, present from the first steps, and its narrowing, which in this run cannot be told apart from the decay of the learning rate.

* The v2.0 hybrid was trained with the old surrogate: matching it does not show that the spiking inputs cost nothing.

* In-context binding still fails: v2.2 answers `volcanoes` for the position of the object and for the word, not for the name.

## v2.1.0: spiking residual highway (2026-10-21)

The projections of every block now read spikes, and the decay of each head can follow the tokens. The repository is now organised by version; the v2.0 scripts, the v2.0 model and its published weights are unchanged.

### Added

* **`v2.1.0/model.py`, the spiking residual highway.** The inputs of `W_qkv`, `W_proj` and `W_up` are coded into signed FS spikes (`fs_in`, `fs_o` after `norm_o`, `mlp.fs_in`), as the input of `W_down` already was. Each branch normalises and codes its own input; the residual stream stays a real-valued accumulator. The output head still reads floats.

* **Data-dependent decay** (`DataDecay`, `DataDecaySpikingSelfAttention`): `γ_t,h = exp(-softplus(w_h · s_t + b_h + β_h))`, computed from the spike input of `W_qkv` with 576 x 9 weights per block. `β` holds the fixed ladder and `w`, `b` start at zero, so at initialisation the model is the one with the fixed ladder. The parallel form uses the cumulative sum of `ln γ`; the recurrent form keeps the state of v2.0, `S_t = γ_t S_{t-1} + k_tᵀ v_t`, `Z_t = γ_t Z_{t-1} + 1`.

* `data_decay` in `CONFIGS` and in `FSGPT` chooses the fixed ladder or the data-dependent decay. `CONFIGS` lists the three runs of the results under their names; at small scale each one gives the same weights and history, bit for bit, as the script it was trained with.

* **Three runs**, with their histories in `v2.1.0/results/history`:
  * fixed ladder, K=2: validation loss 3.7542, perplexity 42.70;
  * data-dependent decay, K=2: 3.6991, perplexity 40.41;
  * data-dependent decay, K=3 for every FS neuron, one K for the whole model, with the same threshold scales as the K=2 runs: **3.6573**, perplexity **38.75**.

* `v2.1.0/diagnostic.py`, the diagnostic of v2.0 for the three v2.1.0 histories. It tells the runs apart by their exact names, which all carry `g var` and `alpha app` and fell into one category under the rules of v1, and compares them against the fixed-ladder run when the folder holds no dense control. It reads `results/history` and writes `figures/` next to itself, from any working folder and as a notebook cell.

* `v2.1.0/generate.py` and `v2.1.0/recurrent_check.py`, self-contained like those of v2.0: they carry the classes of `v2.1.0/model.py` and take the K=3 checkpoint from Hugging Face (`HF_FILE`, `v2.1.0/model.pt`) or from Drive. The recurrent form comes from the classes themselves (`set_recurrent`, `generate_recurrent`); the four checks and the prompt sets of v2.0 are unchanged.

* **Verified on the published K=3 checkpoint** by `v2.1.0/recurrent_check.py`, with no retraining: in float64 the parallel and the recurrent forms give identical logits, with the same argmax on 2 x 1024 tokens; every attention output goes through a spike read, which absorbs the rounding differences. In float32 83.6% of the tokens keep the same argmax, against 95.2% in v2.0, because v2.1 has more spike thresholds a rounding can cross; the validation loss is the same (3.69110 against 3.69117).

* **Qualitative probes** of the K=3 checkpoint in the README, next to those of v1: outputs of `v2.1.0/generate.py`, seed 1234, one completion per prompt. The plural agreement and the domain prose hold; the binding of Alvarez to volcanoes and the singular agreement across "along with the students", both passed by v1, now fail; induction, arithmetic and code fail as in v1.

* `v2.1.0/figures/validation.png`, with a dark variant: the validation curves of the three runs and of the v2.0 hybrid.

### Changed

* **The repository is organised by version.** `v2.0.0/` holds the scripts of v1 and v2.0 (`model.py`, `diagnostic.py`, `generate.py`, `recurrent_check.py`), unchanged, with their results and figures; `v2.1.0/` holds the v2.1 script with its histories and figure. README, CHANGELOG, LICENSE and `docs/` stay at the root.

* The README is rewritten for v2.1 and the v2.0 README moves to `docs/v2.md`. Its text is unchanged; its links and the paths in `docs/v1.md` follow the new layout.

* Training with the data-dependent decay runs at 8 sequences per forward pass and 8 accumulation steps, the same 65,536 tokens per step: the decay matrix is one per sequence, 341 MB of activations per sequence and block at 1024 tokens against 232 MB.

* `v2.1.0/model.py` resumes a run from the update of the step its checkpoint was saved at, without evaluating that step again, and stops instead of resuming when the history records a best that the checkpoint does not have.

### Known limits

* The output head reads floats: 576 x 50,257 MACs per token.

* One seed per run.

* In-context binding and long-range agreement regress against v1 in the qualitative probes: the model takes the most recent candidate.

## v2.0.0: FS²-SSA (2026-10-06)

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
