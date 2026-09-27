# How CLM-8B was built and trained: code-level review

Reviewed 26 September 2026. Purpose: take what is sound from CLM into s1nav and avoid what is not. Each item is marked with its source:
- [code] read in the released repo;
- [ckpt] read from the released checkpoint file;
- [blog] the authors' Notion post;
- [card] a Hugging Face model or dataset card.

## 1. Sources and what is released

| Source | Content | Reviewed |
|---|---|---|
| github.com/Contrastive-LM/CLM (main, fetched 26 Sep) | Serving (`src/clm`), head fine-tuning (`train/finetune.py`, `adapters.py`, `embed_utils.py`), embedding download (`preprocessing/hf_embeddings.py`), best-of-N evaluation (`evaluation/bon_eval.py`), T-Rex benchmark | All training, serving and evaluation code read |
| HF `Contrastive-LM/CLM-v0.1-8B` | Model card, `config.json`, `CLM_v0.1-8B.pt` (75.6 MB) | Checkpoint unpickled without executing code; config and shapes read |
| HF `Contrastive-LM/deepswe-clm-train-embeddings-8k` | Frozen embeddings for DeepSWE head fine-tuning (MIT) | Card read |
| HF `Contrastive-LM/CLM-v0.1-Pretrain-Nemotron` | About 1.02 TB, no card | Metadata only |
| HF `LocalLLaMA/typed-decisions` | Small typed-question benchmark used by the `choice` fine-tuning task | Card read |
| contrastive-lm.notion.site (posted 23 Sep 2026) | Architecture, data recipe, scaling laws, latency details | Full text read |
| arXiv:2602.12281 (CoVer-VLA) | Same group's earlier contrastive verifier for robot actions | Abstract and search excerpts only |

**Not released:**
- pre-training and mid-training code;
- the hard-negative generation prompts;
- post-training data mixing code;
- the scaling-law fits (figures only).

Only the head fine-tuning stage can be reproduced from the repo. The three-stage recipe is described but cannot be re-run.

## 2. Architecture

- **Encoder.** Frozen `Qwen/Qwen3-8B`. This is the generative model, not an embedding-tuned variant. It is served by vLLM with `--runner pooling`, using the hidden state of the final token (last-token pooling), L2-normalised. [code, card]
- **Heads.** Two separate heads, one for states and one for actions, with the same shape: 4096 → 1536 → 1536 → 512, GELU, LayerNorm on the hidden layer, no residual. [ckpt: `width 1536, depth 3, layernorm true, projection_dim 512`]
  - Total trainable parameters for both heads together: 18,887,680. [ckpt] The "20M-parameter head" in the blog is the two heads combined.
- **Score.** exp(logit_scale) × cosine(state projection, action projection), with the scale clamped at 100. A softmax over a question's candidates is the answer. [code]
- **Learned temperature.** The released checkpoint stores logit_scale = 4.613, i.e. exp = 100.8. That is above the clamp, so the effective temperature is the ceiling value τ = 0.01. [ckpt] The model is therefore trained to be as sharp as the clamp allows, and its probabilities are not calibrated by construction.
- **Typed questions** on top of the ranking primitive: [code, `schema.py`]
  - The state text is `context + blank line + question instructions`. The question sits on the state side, so each question needs its own state embedding. The shared context is reused only through vLLM prefix caching.
  - **Choice:** the candidates are the option descriptions (or keys).
  - **Score:** the candidates are the ordered levels; the answer is the expected level index.
  - **Yes/no:** a two-way softmax between `"true: Yes. This is true: <q>"` and `"false: No. This is false: <q>"`.
  - The "confidence" field is top probability minus the mean of the other probabilities: a heuristic, not a calibrated quantity.
- **Why it is fast.** Action embeddings are cached (a vector store on the GPU), so a fixed action set costs one state embedding plus dot products per decision. Measured on a 200-token state on an RTX 4090: 36 ms with 1 candidate, 44 ms with 1,024 candidates. [blog]

## 3. Embedding recipe

- **Training-time token recipe.** [code, `embed_utils.py`]
  - A chat-message state is rendered with the chat template; the last `max_len − 1` tokens are kept.
  - A string state is tokenised without special tokens; the tail is kept.
  - An action is tokenised without special tokens; the head is kept.
  - `max_len` is 8192 for DeepSWE and 2048 for typed questions.
- **Serving-time path.** Raw text is sent to the vLLM `/v1/embeddings` endpoint with `truncate_prompt_tokens` = 2048. [code, `embedder.py`] This is a different code path from training, so train/serve skew is possible: special tokens, chat template versus plain text, and truncation side. The repo has no test that checks the two paths agree.
- **Storage.** Embeddings are stored as float16, 4096-d, about 8 KB per text. [code]

## 4. Training

### 4.1 Pre-training, mid-training and post-training (described; code not released)

**Stage 1: pre-training.** [blog]
- About 60M Nemotron DQA question–answer pairs (question = state, answer = action).
- Bidirectional in-batch InfoNCE.
- The encoder is frozen and its embeddings are precomputed once, so a full pre-training run takes about 1 hour on one RTX 4090.

**Stage 2: mid-training.** [blog]
- About 30M hard negatives generated by Gemini 2.5 Flash-Lite: semantically similar but wrong answers for a subset of the questions.
- They are added to the state-to-action denominator.
- Measured on about 100k held-out questions (1 gold answer, 10 hard negatives):

  | Recipe | Top-1 accuracy |
  |---|---|
  | Pre-training only | 52.1% |
  | Pre-training, then a short hard-negative stage | 69.2% |
  | Hard negatives from the start | Peaked at 62.4%, then overfit |

- Their conclusion: hard negatives work best as a refinement after broad training.

**Stage 3: post-training.** [blog]
- About 1M agent trajectory steps (ADP, Endless-Terminals, LiteCoder-Terminal-SFT), each step a (context, decision) pair.
- The mixture is 40% Nemotron replay and 60% agent data.
- Hard-negative top-1 was 69% before post-training; 68.5% after with replay; 56.2% after without replay.

**Scaling.** [blog] Test InfoNCE loss follows power laws in compute, data, head size and encoder size; the encoder size helps most. The optimal head size grows linearly with data, at about 310 training tokens per head parameter.

### 4.2 Head fine-tuning (released code, `train/finetune.py`)

**Loss.**
- Bidirectional in-batch InfoNCE.
- **False-negative masking:** other samples with the same (task, step) key are masked out of each row's negatives, so parallel rollouts at the same step do not count as negatives.
- A `choice` variant trains against soft targets (annotator probability distributions) or hard labels.

**Batching.**
- `random` sampling, or `task-blocked`: 4 tasks per batch, which makes in-batch negatives harder.
- Default batch 2048 (DeepSWE) or 256 (choice).

**Optimisation.**
- AdamW, weight decay 0, gradient-norm clip 1.0.
- One-cycle schedule with cosine annealing, 10% warm-up.
- Learning-rate rule: lr = 2e-3 × √(1024/width) × √(batch/1024). That gives about 2.3e-3 at width 1536 and batch 2048.
- Initial logit scale log(1/0.07), learned, clamped at 100.
- 20 epochs, patience 5.

**Splits and model selection.**
- Held-out tasks are removed first.
- A task-disjoint validation split (10%) is drawn from the rest.
- Optional K-fold task-disjoint cross-validation, stratified by candidate count and pass count.
- The checkpoint is selected by within-task top-1 on the validation tasks.

**Outcome labels are not used in the loss.**
- The DeepSWE training data contain every step of every rollout (405,919 steps, 4,701 trajectories, 113 tasks), passing and failing. [card]
- In the released code the `reward` column is used only to stratify folds. [code]
- The "verifier" is therefore trained to match actions to their states, not to separate good actions from bad ones. At evaluation, a trajectory's score is the mean state–action similarity over its last 12 steps, and the highest-scoring of N candidates is chosen. [code, `bon_eval.py`] That this picks passing trajectories 31/38 times is an empirical result. The training objective does not target it.

## 5. Evaluation practice

**What is sound:**
- held-out tasks are separate from the validation tasks;
- folds are stratified;
- best-of-N uses an exact expectation over random N-subsets with uniform tie-breaking;
- random-pick and oracle-any baselines are reported. [code]

**Weak points:** [code, results json, figures]
- Small samples: 38 and 30 held-out tasks; 5 runs for each game.
- No intervals.
- Latency compares a local GPU against Jev's public API.
- The T-Rex tie relies on the harness's safety shield: 4,883 interventions for CLM against 28 for Jev; agreement with the planner 0.658 against 0.987.
- No calibration measurement anywhere in the repo.

**Third-party calibration data point.** The `LocalLLaMA/typed-decisions` card (400 test cases, 2,000 decisions, synthetic) reports Jev 1.13.0 at:

| Model | Accuracy | KL from gold | Brier |
|---|---|---|---|
| Jev 1.13.0 | 0.727 | 1.442 | 0.148 |
| Input-ignoring prior | — | 0.347 | 0.189 |
| ModernBERT specialist | — | — | 0.119 |

The benchmark is small, synthetic, and hosted alongside a vendor's claim for its own model, so treat it as weak evidence. It is still the only external calibration number we found for Jev, and it does not support strong calibration.

## 6. Related work from the same group

CoVer-VLA (arXiv:2602.12281, Stanford and NVIDIA) is a contrastive verifier for robot actions.
- According to a third party that used it: SigLIP2 text encoder, ViT-L vision encoder, and a 4-layer transformer over a 10-step action history. It is trained on BridgeV2/DROID augmented with rephrased instructions.
- It generates rephrased instructions and several action candidates, then verifies them.
- It reports larger gains from scaling verification than from scaling policy training on the same data.
- The CLM blog calls it the "robotics variant" of CLM.
- A multimodal CLM-35B is announced for early October 2026.

## 7. What s1nav takes, and what it does not

Details are in `s1nav_spec.md` v0.2, whose changes list maps each item to a section.

| CLM practice | s1nav decision | Why |
|---|---|---|
| Frozen backbone, embeddings precomputed once, head-only training | **Adopt** as stage S1f and baseline B7, before LoRA training | Head experiments drop from days to minutes; it measures what LoRA adds |
| Raw-embedding ablation (`clm-raw`) | **Adopt** as baseline B8: a linear probe on frozen features | A floor for what the heads contribute |
| 3-layer head, width 1536, GELU, LayerNorm | **Adopt** as the head default (shared trunk, per-head outputs) | Tested design at this embedding width |
| Head optimiser recipe (AdamW, wd 0, clip 1.0, one-cycle 10% warm-up, lr rule, large batch) | **Adopt** for head-only training | Cheap to follow; a known starting point |
| Two-stage hard negatives (broad training, then short refinement) | **Adopt** as stage S1b for the `arrive` head, with 40% replay | Their ablation favours it (69.2% vs 62.4%) |
| Replay to prevent forgetting, tracked on a fixed probe | **Adopt**: fixed near-miss probe set tracked across all stages | Detects forgetting between S1, S1b and DAgger |
| Task-disjoint validation, stratified folds, pre-set selection metric | **Adopt** (scene-disjoint calibration split, stratified; pre-registered selection metric) | Matches our leakage rules |
| Scaling ladder | **Adopt** a small data-scaling curve on frozen features (25/50/100%) | Decides whether more data or the 27B model is worth it |
| Separate state and action encoders with action caching | **Do not adopt** | 10 fixed macros read in one pass; nothing to cache |
| Learned temperature as the source of probabilities | **Do not adopt** | CLM's scale sits at the clamp; s1nav fits temperatures post hoc on held-out scenes |
| "Confidence" = top minus mean of the rest | **Do not adopt** | Report probabilities only |
| Verifier trained on alignment only (reward unused) | **Do not adopt**; any plan scorer is trained on observed outcomes | The escalation logic needs outcome probabilities |
| Separate train and serve text paths | **Guard against**: one `build_stream`, plus a feature-cache equivalence test | Avoids the skew risk noted in section 3 |
