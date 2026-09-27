# s1nav: a fast, calibrated navigation model for VLN-CE

Build specification, version 0.2 (26 September 2026). For implementation by Claude Code.

### Changes in v0.2 (from the CLM-8B review, `docs/clm_build_review.md`)

1. New stage **S1f**: frozen-backbone features at every decision are extracted once, and the heads are trained on them alone. It serves as a fast experiment loop and as baselines B7/B8 (Sections 7.1, 7.4, 10).
2. Head default changed to a shared 3-layer trunk (hidden → 1536 → 1536, GELU, LayerNorm) with per-head output layers (Section 5.4).
3. New stage **S1b**: a short hard-negative refinement for `arrive`, with 40% replay. Hard-negative mining is defined in Section 6.3.
4. A fixed near-miss probe set is tracked across all stages to detect forgetting (Sections 6.3, 11.1).
5. Pre-registered checkpoint-selection metric and a stratified calibration split (Sections 6.1, 7.2).
6. Data-scaling curve on frozen features (M3), and new ablations A6–A8 (Section 12).
7. Calibration rules made explicit: no learned temperature inside the heads and no "confidence" heuristic (Section 8).
8. Optional outcome-trained plan scorer for System 2 proposals (Section 9.4).
9. New tests for feature-cache equivalence and probe-set integrity (Section 15.2).

## 1. Purpose

Build a vision-and-language navigation (VLN) model in the style of TypeSafe's Jev. At each step it returns typed decisions with calibrated probabilities in one forward pass, and generates no text. It runs in the standard R2R-CE setting (Habitat simulator, Matterport3D scenes, monocular RGB, continuous movement). When its own probabilities say it is unsure, it hands control to a slower reasoning model ("System 2").

The build has three deliverables:

1. The model: a Qwen vision-language backbone with LoRA adapters and small readout heads.
2. A simulation harness for training data, closed-loop evaluation and the System 2 hand-off.
3. An evaluation suite that measures navigation success, calibration and the cascade trade-off, all in simulation.

Working name: `s1nav` (System One navigator).

## 2. Design summary

- **Backbone.** Develop on Qwen3.5-9B. Move to Qwen3.8-27B only after the 9B model passes gate G3 (Section 12). Both use the same model class and the same vision tower (Section 3).
- **One forward pass per step.** The input is an interleaved stream: instruction, then for each step a frame, a readout token `<act>` and the action taken. Heads read the hidden state at `<act>`. The stream persists across steps, so each step costs only the new frame's tokens.
- **Five heads (the "questions"):**
  - `move`: Choice over 10 movement macros.
  - `arrive`: probability the agent is within 3 m of the goal.
  - `progress`: 10-bin distribution over the fraction of the path completed.
  - `stuck`: probability the last forward move failed.
  - `value`: probability the episode will succeed from here.
- **STOP decision.** STOP is decided by `arrive`, not by `move`. Section 5.4 gives the rule.
- **Training.** Heads first trained on frozen-backbone features (S1f), then behaviour cloning with LoRA on simulator-oracle labels (S1), a short hard-negative refinement (S1b), then DAgger rounds. The `arrive` and `value` heads get outcome labels from the simulator. This is the supervised form of TypeSafe's stated RLCD objective: a proper scoring rule against real outcomes. Temperature scaling runs on a held-out set of training scenes.
- **Cascade.** A controller escalates to System 2 when head probabilities cross thresholds. System 2 is either Qwen3.8-27B in thinking mode (local, same weights with the adapter off) or GPT-6-Astra (paid API).
- **Evaluation.** Everything runs in Habitat: full R2R-CE val-unseen (1,839 episodes), the 100-episode subset used by the GPT-6-Astra report, RxR-CE, and a perturbation suite.

## 3. Facts this spec relies on

Each fact was checked on 26 September 2026. Claude Code should re-check any marked "re-verify at M0" before relying on it.

| Fact | Value | Source | Status |
|---|---|---|---|
| Qwen3.8-27B model class | `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5` | HF `Qwen/Qwen3.8-27B` config.json | Verified |
| Qwen3.8-27B size and layout | 27.78B params; 64 layers (48 linear-attention Gated DeltaNet, 16 full attention; full attention every 4th layer); hidden 5120; 1 MTP layer | config.json, HF API | Verified |
| Qwen3.5-9B size and layout | 9.65B params; 32 layers (24 linear, 8 full); hidden 4096 | config.json | Verified |
| Vision tower (both models) | depth 27, width 1152, patch 16, spatial merge 2, so one visual token per 32×32 pixels. `out_hidden_size` differs (4096 vs 5120), so adapters and heads do not transfer between the two models. | config.json | Verified |
| Minimum image size | `shortest_edge` 65,536 pixels (256×256, 64 visual tokens) | preprocessor_config.json | Verified |
| Licence | Apache-2.0 (both models) | HF API | Verified |
| Qwen3.8-27B BF16 / FP8 weight size | 55.6 GB / 30.9 GB | Published file sizes | Reported by third party |
| DGX Spark | 128 GB unified memory, 273 GB/s bandwidth, about 99.8 TFLOPs BF16 and 207.7 TFLOPs FP8 measured | NVIDIA spec; StorageReview | Reported |
| habitat-sim 0.3.3 conda builds | linux-64 and osx-arm64 exist. **No linux-aarch64 build**, so the DGX Spark has no prebuilt package. Python 3.9 only. | conda.anaconda.org/aihabitat repodata | Verified |
| Headless rendering | Relies on EGL, which does not work on macOS. A Mac needs a windowed context. | habitat-sim README | Reported; re-verify at M0 |
| habitat-lab main | `VLN-v0` task, `R2RVLN-v1` dataset loader, `ShortestPathFollower`, and `benchmark/nav/vln_r2r.yaml` (forward 0.25 m, turn 15°, 500 steps, RGB 256×256, HFOV 90) | habitat-lab GitHub main | Verified |
| VLN-CE reference task | Success 3.0 m; `ALLOW_SLIDING: True`; nDTW ground-truth paths in `R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz`; DAgger defaults p=0.5, 10 iterations; MIT licence; pinned to habitat-sim 0.1.7 | VLN-CE repo | Verified |
| StreamVLN trajectory data | Gated HF dataset, about 131 GB, CC-BY-SA-4.0, rendered with habitat-sim 0.2.4 | HF API | Verified |
| MP3D scenes | Require signing the Matterport Terms of Use (non-commercial research) | Matterport project page | Known; user action |
| CLM-8B reference head | Two heads, each 4096 → 1536 → 1536 → 512, GELU, LayerNorm, 18.9M parameters in total; learned logit scale exp = 100.8 (at the clamp of 100) | `CLM_v0.1-8B.pt` read directly; repo `train/finetune.py` | Verified |
| CLM two-stage hard negatives | Pre-training only 52.1%; pre-training then short hard-negative stage 69.2%; hard negatives from the start peaked at 62.4%. Replay kept hard-negative top-1 at 68.5% (56.2% without) | CLM blog, 23 Sep 2026 | Reported by the authors; code not released |
| GPT-6-Astra API price | $10 per million input tokens, $50 per million output | Passing mention in Yotta Labs, "Qwen 3.8 27B: Specs, Hardware Requirements, and How to Run It (2026)"; not an OpenAI price page | Reported by third party; re-verify before any paid run |

## 4. System architecture

### 4.1 Processes

```
┌──────────────────────── Mac (M5 Max) or x86 GPU box ────────────────────────┐
│  envserver  (Python 3.9, habitat-sim 0.3.3, habitat-lab 0.3.x)              │
│   - episode control, rendering, macro expansion, metrics                    │
│   - privileged oracle endpoints (train mode only)                           │
│   - records every episode to a per-episode log                              │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │ RPC over LAN (msgpack over HTTP or gRPC; JPEG frames)
┌───────────────┴────────────────── DGX Spark ─────────────────────────────────┐
│  policyd   (Python 3.12, torch, transformers, peft)                         │
│   - s1nav model: backbone + LoRA + heads, persistent stream cache           │
│   - escalation controller                                                   │
│   - System 2 adapter: same backbone with LoRA disabled, thinking mode;      │
│     or remote GPT-6-Astra via the MCP adapter                               │
│  trainer   (same env as policyd)                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

The simulator and the model must run in separate processes. habitat-sim 0.3.3 is built only for Python 3.9, and the model stack needs a current Python.

### 4.2 Hardware placement

1. **Default.** Run envserver on the Mac from the osx-arm64 conda build, with a logged-in GUI session for the GL context. Run policyd and trainer on the Spark. Rendering throughput on the Mac is unknown; M0 measures it.
2. **Alternative.** Build habitat-sim from source on the Spark (aarch64, headless EGL, CUDA) so everything runs on one box. M0 attempts this with a fixed time box. There is no prebuilt package, so treat success as uncertain.
3. **Fallback for large rendering or evaluation runs.** Rent an x86 Linux GPU host and use the linux-64 headless conda build.

Everything that affects rendering must be recorded in the run manifest (Section 15.3): host, OS, habitat versions, GL renderer string. Results from different renderers are not pooled until V0 shows they agree.

## 5. Model

### 5.1 Backbone and adaptation

- Backbones:
  - Dev: `Qwen/Qwen3.5-9B`.
  - Final: `Qwen/Qwen3.8-27B`.
  - Pin both by Hugging Face commit hash.
- Vision tower frozen in v1. Ablation A4 unfreezes its last 4 blocks.
- LoRA on the language model:
  - Rank 64, alpha 128, dropout 0.05.
  - Targets: attention q/k/v/o, MLP gate/up/down, and the Gated DeltaNet input and output projections. Claude Code should list the module names from the loaded model and record them in the manifest. Do not guess the names.
- Heads are trained in full. The MTP layer is unused and stays frozen.
- Precision: BF16 weights and compute. QLoRA (4-bit base) is a fallback only if 27B BF16 does not fit (M1 measures this). Keep the training and inference quantization the same, because calibration is fitted to one numeric path.

### 5.2 Sequence format

One episode is one stream. At step t:

```
[system header][instruction text]
<frame_1><act> a_1 <frame_2><act> a_2 ... <frame_t><act>
```

- `<frame_t>` is the standard Qwen vision block for one RGB image (vision start token, image tokens, vision end token).
- `<act>` is one new special token added to the tokenizer, with a trainable embedding. Heads read the final-layer hidden state at each `<act>` position.
- `a_t` is the macro actually executed, written as a short fixed string (e.g. `F3`, `L2`, `R1`). It is appended after the heads are read, so the model sees its own action history. At training time `a_t` is the executed action from the log. Under DAgger that can be the student's action; the loss target is always the oracle label.
- During escalation, System 2's executed actions are written into the stream the same way, with a different fixed prefix (`S2:F3`). This makes hand-offs visible to the model.

Frame resolution: default 384×384 (144 visual tokens). Ablation A1 compares 256×256 (64 tokens) and 512×512 (256 tokens). Render at the chosen resolution; do not resize at the model.

### 5.3 Streaming and windowing

- Inference keeps a persistent hybrid cache: key-value cache for the 16 full-attention layers, recurrent and convolution state for the 48 DeltaNet layers. Each step appends one frame block and `<act>`, reads the heads, then appends `a_t`. The `<act>` token stays in the stream, so no cache rollback is needed.
- Windowing. When the stream would exceed `L_max = 8192` tokens, rebuild it from the header, the instruction and the last `K = 24` steps (frames and actions), then continue. Training uses the identical rebuild on long episodes.
  - **Requirement:** one function, `build_stream(episode_log, t, L_max, K)`, used by both the trainer and policyd. A unit test asserts that the two produce identical token ids.
- Equivalence test. For 20 logged episodes, the head outputs from incremental cached inference must match a single full-sequence forward pass within tolerance (max absolute logit difference below 1e-2 in BF16). If the hybrid cache API cannot do this, stop and report. Do not paper over it.

### 5.4 Heads, labels and the decision rule

The heads share one trunk on the `<act>` hidden state: hidden → 1536 → 1536, GELU, LayerNorm on the hidden layer (the CLM-8B head shape). Each head has its own linear output layer from the 1536-d trunk. That is about 8.7M parameters for 9B (hidden 4096) and 10.2M for 27B (hidden 5120).

Defaults and ablations:
- A single readout token is the default; ablation A3 uses one token per head.
- Ablation A8 uses a separate trunk per head, and a 2-layer trunk.
- The heads output logits with no learned temperature or scale parameter. Temperatures are fitted after training (Section 8).

| Head | Type | Output | Label source (simulator) | Loss |
|---|---|---|---|---|
| `move` | Choice | softmax over 10 macros: `F1 F2 F3 F4` (0.25–1.0 m), `L1 L2 L4`, `R1 R2 R4` (15°, 30°, 60°) | Oracle primitive sequence from the current state, collapsed into the largest macro that fits (5.5) | Cross-entropy |
| `arrive` | Bool | p(geodesic distance to goal < 3.0 m now) | Geodesic distance from the navmesh pathfinder | Binary cross-entropy |
| `progress` | Score | softmax over 10 bins of path fraction completed | 1 − (geodesic to goal / geodesic from start), clipped to [0, 1] | Cross-entropy |
| `stuck` | Bool | p(last forward macro moved < 5 cm per primitive) | Measured displacement | Binary cross-entropy |
| `value` | Bool | p(episode ends in success from this state under the current policy) | Final outcome of the same episode; on-policy rollouts only (β = 0) | Binary cross-entropy |

Loss: sum of per-head losses with weights move 1.0, arrive 1.0, progress 0.5, stuck 0.25, value 0.5 (starting values, tunable). Label smoothing and class re-weighting are off: both distort probabilities, and calibration is a primary output.

Decision rule at each step:

1. If `p_arrive ≥ τ_stop`, issue STOP.
2. Otherwise execute the argmax of `move`.
3. At the 500-primitive budget, the episode ends (not a STOP).

`τ_stop` is fitted on the calibration split (Section 8). Ablation A2 puts STOP back into `move` as an 11th option for comparison.

### 5.5 Action space and macro expansion

- Primitives are the standard VLN-CE set: STOP, forward 0.25 m, turn left 15°, turn right 15°. Metrics and the 500-primitive budget always count primitives, so results compare with the literature.
- envserver expands a macro into primitives and executes them in order. If a forward primitive moves less than 5 cm, the rest of the macro is abandoned. The response reports how many primitives were executed and the displacement. This abort rule is a harness setting (`abort_on_block: true`) and is recorded in the manifest.
- Oracle-to-macro rule: take the oracle's primitive sequence from the current state, find the leading run of identical primitives (n), and emit the largest macro of that type with size ≤ n. A round-trip unit test checks that expanding the macro labels reproduces the oracle path.
- Sliding along walls: `ALLOW_SLIDING = True` by default, matching VLN-CE. V5 also evaluates with sliding off.

### 5.6 Oracle

- Primary oracle: `ShortestPathFollower` to the episode goal on the navmesh, matching the VLN-CE teacher.
- Secondary oracle: shortest path to the next reference-path waypoint, advancing when within 0.5 m.
- For every training episode, log the nDTW of each oracle against the reference path. If they diverge (primary-oracle nDTW < 0.8), use the secondary oracle's labels for that episode and flag it. The share of flagged episodes is reported at M0.

## 6. Data

### 6.1 Sources and splits

| Split | Use | Notes |
|---|---|---|
| R2R-CE v1-3 `train` minus calibration scenes | BC and DAgger training | 61 MP3D training scenes |
| Calibration split: all `train` episodes from 5 held-out training scenes (fixed list, seed 0) | Temperature scaling, τ thresholds, early stopping | Never used for gradients |
| R2R-CE `val_seen` | Overfitting gap only | |
| R2R-CE `val_unseen` (1,839 episodes, 11 scenes) | Primary closed-loop evaluation | Never used for any fitting |
| R2R-CE-100 (Open-Nav's 100 val-unseen episodes) | Comparison with the GPT-6-Astra report | Obtain the episode-ID list from the Open-Nav release; if it cannot be found, record that and skip |
| RxR-CE `val_unseen`, English guide instructions | Language generalization | Same scenes as R2R-CE val-unseen; longer instructions |
| EnvDrop-augmented R2R-CE episodes | Optional extra BC data for the 9B model | From the VLN-CE release; train scenes only |
| StreamVLN trajectory data | Optional; not used in v1 | Rendered with habitat-sim 0.2.4 and possibly different camera settings, so matching must be shown first |

Guard: the loader checks that no episode used for training or calibration comes from a val-unseen scene. This runs as a test and again at the start of every training job.

### 6.2 Rendering

- Camera: RGB only, 384×384 (per 5.2), HFOV 90°, habitat-lab default sensor height. Record the actual values.
- Oracle rollouts (BC data):
  - For each training episode, roll out the primary oracle and log per step: frame (JPEG, quality 90), executed macro, oracle macro, all head labels, position and heading.
  - Also log a "noisy oracle" variant: with probability 0.1 per step, execute a random non-oracle macro and label the following states with the oracle. This gives some off-path states before DAgger.
- Storage: WebDataset tar shards, one sample per episode (JSON record plus frames). Estimated size at 384×384 is tens of GB for R2R-CE train; M0 measures it.
- Privileged fields (position, heading, geodesic distances) are stored for label generation and analysis. The model's input builder must not read them. A test asserts that the input builder's accessed fields are a whitelist.

### 6.3 Hard negatives and the near-miss probe set

CLM found that hard negatives work best as a short refinement after broad training, and that replay keeps earlier skills from being lost (`clm_build_review.md`, Section 4.1). s1nav applies this to the `arrive` head. That head is where the GPT-6-Astra report's false stops occur: for example, stopping beside a fireplace in the wrong room.

**Mined states** (from training scenes only, rendered like other training data, labelled by the simulator):

1. **Distance near-misses.** States 3–8 m (geodesic) from the goal with path progress ≥ 0.7. Label `arrive = 0`.
2. **Landmark near-misses.** States in a different region from the goal whose visible objects include the instruction's final landmark.
   - The landmark is extracted once per instruction with the local Qwen model (e.g. "stop at the refrigerator" → `refrigerator`) and mapped to an MP3D object category.
   - Regions and objects come from the MP3D semantic annotations in Habitat. Claude Code must confirm at M0 that the region and object annotations load for all training scenes. If they do not, drop type 2 and mine only types 1 and 3, which do not need the annotations, and record that.
   - Label `arrive = 0`.
3. **Passed-then-left.** From noisy-oracle and DAgger rollouts: states after the agent has been inside 3 m and then moved beyond it. Label `arrive = 0`.
4. **Matched positives.** For each negative, a state within 3 m of the same goal, so the stage does not shift the base rate. Label `arrive = 1`.

**Near-miss probe set.**
- Built once, from the calibration scenes only: 1,000 mined negatives (types 1–3 in equal parts) and 1,000 matched positives. Seed 0.
- Frozen after M1 and never used for gradients.
- Tracked after every stage (S1f, S1, S1b, each DAgger round, S3): `arrive` AUROC, accuracy at τ_stop, and ECE.
- It plays the role of CLM's fixed hard-negative accuracy check: a drop of more than 3 AUROC points between stages is flagged as forgetting.

**Calibration split stratification.** The 5 held-out calibration scenes (6.1) are chosen by stratified draw over training scenes, binned by navigable area and episode count (seed 0), not by a plain random draw. This keeps the calibration split typical of the training distribution.

## 7. Training

### 7.1 Stages

| Stage | What | Data | Notes |
|---|---|---|---|
| S0 | Pipeline smoke test | 50 training episodes | 9B must reach > 95% `move` accuracy on these 50 (overfit test) |
| S1f | Frozen-feature head training | Oracle and noisy-oracle rollouts, features extracted once (7.4) | Heads only. This is the fast experiment loop for head design, loss weights and data mixes, and gives baselines B7/B8. |
| S1 | Behaviour cloning with LoRA | Oracle and noisy-oracle rollouts | 2 epochs; heads initialised from the best S1f heads; `value` head masked (no on-policy labels yet) |
| S1b | Hard-negative refinement | 60% mined states (6.3) and 40% replay of S1 data | Short: at most 0.25 epoch-equivalents of S1. Stop early if probe AUROC stops improving. Ablation A6 compares this against mixing hard negatives in from the start of S1, and against none. |
| S2 | DAgger, 4 rounds | Aggregated rollouts | Round r runs a per-decision mixture: oracle with probability β_r ∈ {0.5, 0.25, 0.1, 0}, else student argmax. Collect about 5,000 episodes per round from training scenes. Label every visited state with the oracle. Fine-tune from the previous checkpoint on the aggregate. |
| S3 | Calibration | Calibration split | Section 8 |
| S4 (optional) | Teacher distillation | States where the student is uncertain | Section 9.3; only if G3 is passed and budget allows |

DAgger rounds keep the mined near-miss states in the aggregate at 10% of samples. `value` labels come only from β = 0 episodes, which starts at round 4. Also add dedicated on-policy collection: after round 3, run 2,000 extra β = 0 episodes for `value` labels. Weight value samples from the latest policy 1.0 and from older rounds 0.5.

### 7.2 Hyperparameters (starting values)

- Optimizer: AdamW, β = (0.9, 0.95), weight decay 0.
- Learning rate: LoRA 1e-4, heads and `<act>` embedding 1e-3. Cosine schedule, 3% warm-up.
- Effective batch: 16 episodes (gradient accumulation as needed). Maximum sequence length `L_max` = 8192.
- Gradient checkpointing on.
- Seeds:
  - 9B ablations: 3 seeds.
  - 27B main run: 1 seed, stated as such in every report.
- **Checkpoint selection (pre-registered).** Pick the checkpoint with the lowest mean negative log-likelihood of `move` and `arrive` on the calibration split. Record probe AUROC but do not use it for selection. Never use val-unseen for any selection.

### 7.3 Throughput requirement

M1 measures tokens per second for 9B and 27B on the Spark, for the actual hybrid architecture and sequence lengths. Gated DeltaNet needs specialised kernels to be fast. Claude Code must confirm those kernels load and run on the Spark's aarch64 CPU and Blackwell GPU, and record which kernel path is active. If only a slow reference path works, report the measured throughput before starting S1.

### 7.4 Frozen-feature stage (S1f)

**Extraction.**
- Run the frozen backbone (no LoRA) once over every training and calibration stream and store the final-layer hidden state at each decision point.
- With a frozen backbone the new `<act>` token has no trained embedding, so S1f streams use a fixed plain-text readout marker (`" Next:"`). The readout is the marker's last token. `build_stream(..., readout="text")` produces this variant, and the rest of the stream is identical.
- Features are stored as float16, with episode id, step index, scene id and all head labels. Size: about 8 KB per decision for 9B (4096-d) and 10 KB for 27B.
- Extraction cost is one forward pass: about 2 × parameters FLOPs per token. At 384×384 that is about 5 h for 9B and 15 h for 27B per pass over the BC data, against about 23 h and 66 h for each LoRA epoch (Section 13).

**Head training on cached features.** The CLM-8B head recipe, as starting values:
- AdamW, weight decay 0, gradient-norm clip 1.0.
- One-cycle schedule with cosine annealing and 10% warm-up.
- Peak learning rate 2e-3 × √(1024/width) × √(batch/1024), which is about 2.3e-3 at width 1536 and batch 2048.
- Batch 2048 decisions, 20 epochs, patience 5.
- Batches are drawn scene-blocked: 4 scenes per batch, the analogue of CLM's task-blocked sampler. Ablation against random batches.

**Uses.**
1. Head architecture, loss weights and data-mix decisions are made here first, then confirmed once with LoRA.
2. Baselines B7 (frozen features + heads) and B8 (a linear probe on frozen features for each head).
3. Data-scaling curve (M3): train S1f heads on 25%, 50% and 100% of training episodes (3 seeds each), fit a power law to calibration-split NLL, and report the extrapolated gain from doubling data. This informs whether EnvDrop data or the 27B model is worth their cost.

**Limit.** Cached features are valid only for logged trajectories. Closed-loop evaluation of S1f heads runs the frozen backbone live in policyd, with the adapter off and the heads loaded.

## 8. Calibration

Rules taken from the CLM review:
- The heads have no learned temperature or scale. CLM-8B's learned scale sits at its clamp (effective τ = 0.01), which makes probabilities as sharp as possible. Calibration here comes only from the post-hoc fit below.
- No derived "confidence" score such as top probability minus the mean of the rest. Only probabilities are reported and used by the controller.
- Always report ECE before scaling as well as after.

- Fit one temperature per head (`move`, `progress`, and each binary head) on the calibration split by minimising negative log-likelihood. Store the temperatures in the checkpoint.
- Fit thresholds on the calibration split:
  - `τ_stop` to maximise success rate.
  - Escalation thresholds (Section 9.2) by sweeping.
- Report before and after scaling, on both the calibration split and val-unseen:
  - Expected calibration error (15 equal-mass bins).
  - Brier score and negative log-likelihood.
  - Reliability diagrams.
- For `move`: the risk-coverage curve and its area (AURC), using max probability as the confidence.
- Composition check. Compare `value` at t = 0 with the episode's actual success, and track calibration of `value` at each step index. The goal is to measure how calibration changes over the episode, not to assume it carries through.

## 9. Inference server and escalation

### 9.1 policyd API (for envserver-driven evaluation)

- `start(episode_id, instruction)`: new stream.
- `decide(frame_jpeg)` returns:
  `{move_probs[10], p_arrive, progress_probs[10], p_stuck, p_value, decision, escalate: bool, latency_ms}`.
- `commit(executed_action, source ∈ {s1, s2})`: appends the executed action to the stream.
- `end()`: frees the cache.

Per-step latency is logged. Expected lower bound for one forward pass is weight bytes / 273 GB/s: about 0.07 s (9B BF16), 0.20 s (27B BF16), 0.11 s (27B FP8). M1 measures the real figure.

### 9.2 Escalation controller

Escalate for the next k = 4 decisions (or until System 2 issues STOP) if any of these holds:

1. `max(move_probs) < τ_a` on two consecutive steps.
2. `τ_low ≤ p_arrive < τ_stop` (uncertain arrival).
3. `p_stuck > 0.5` on two consecutive steps.
4. `p_value` drops by more than Δ over the last 5 steps.

The thresholds (τ_a, τ_low, Δ) are swept on the calibration split to trace success rate against the fraction of decisions escalated. Each trigger can be switched on and off for ablation.

### 9.3 System 2

- **Interface.** System 2 is an agent with the same tools as the GPT-6-Astra report: `observe()` returns the current RGB image without spending budget; `step(actions)` executes primitives or macros. It receives the instruction, the last 8 frames and a compact action history. It returns up to k macros or STOP.
- **Adapter.** A small MCP adapter exposes envserver's `observe` and `step` to any LLM agent.
- **Options:**
  - (a) Qwen3.8-27B in thinking mode on the Spark, with the s1nav LoRA disabled (one copy of the weights, adapter toggled).
  - (b) GPT-6-Astra through its API.

  Option (b) costs money per episode. Measure the cost on 10 episodes before any larger run. Check the provider's terms of use before using its outputs as training labels in S4; OpenAI's terms have historically restricted using outputs to develop competing models.

### 9.4 Optional: outcome-trained plan scorer (M5)

CLM's clearest gain was as a best-of-N selector. At an escalation, System 2 can propose N candidate k-step plans (default N = 4), and a scorer picks one.

- **Model.** The candidate plan is appended to the stream as action text after `<act>`, and a new Bool head reads a second readout token: p(the plan increases path progress by ≥ 0.1 within k decisions and does not end in a false STOP).
- **Training.** Train on logged System 2 proposals with their observed outcomes. Unlike CLM's DeepSWE head, whose loss never sees the reward, this head is trained on outcomes. Only proposals that were actually executed have labels, so use the executed proposal per escalation (off-policy data are reported as a limitation).
- **Gate.** Adopt only if it raises SR at a fixed escalation rate on the calibration split.

## 10. Baselines (same harness, same episodes)

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Random macro; forward-only then STOP | Harness sanity (near-zero SR expected) |
| B1 | Primary oracle | Harness ceiling |
| B2 | Qwen3.8-27B zero-shot LLM agent via observe/step, thinking on | Local System 2 on its own. Replicates the GPT-6-Astra report's protocol with an open model. Run on R2R-CE-100 and a fixed 300-episode val-unseen sample. |
| B3 | s1nav without temperature scaling | Isolates the effect of calibration |
| B4 | s1nav 9B vs 27B | Effect of backbone size |
| B5 (optional) | StreamVLN public checkpoint in this harness | Harness check against its reported R2R val-unseen SR 56.4 / SPL 50.2 |
| B6 (optional, paid) | GPT-6-Astra re-run on R2R-CE-100 | Matched comparison with the report's 79.0% |
| B7 | Frozen backbone + trained heads (S1f), run closed-loop | How much LoRA adds over frozen features |
| B8 | Frozen backbone + linear probe per head | Floor for the trunk's contribution (the analogue of CLM's `clm-raw`) |

Published numbers from other papers go in a separate table labelled as not matched: different splits, sensors and harnesses.

## 11. Validation plan (all in simulation)

### 11.1 Stages

| Stage | Question | Test | Pass criterion |
|---|---|---|---|
| V0 Harness | Does the simulator and metric code match the standard task? | B0 and B1 on val-unseen; our SR/SPL/nDTW/NE/OSR versus habitat-lab's own measures on the same trajectories; optional B5 | Metrics agree with habitat-lab to 1e-6 on identical trajectories. B1 SR ≥ 0.95 (expected, not a literature value; if lower, investigate navmesh and episode issues before going on). B0 near zero. |
| V1 Offline | Do the heads learn and are they calibrated on held-out data? | Teacher-forced on the calibration split and on val-unseen oracle rollouts | `move` accuracy, ECE, Brier, NLL, AURC reported |
| V1b Probe | Did a stage cause forgetting or fix false stops? | Near-miss probe set (6.3) after every stage | Probe AUROC reported per stage; a drop of > 3 points is flagged |
| V2 Closed loop | How well does it navigate? | Full val-unseen (1,839); val-seen; R2R-CE-100 | NE, OSR, SR, SPL, nDTW with 95% CIs |
| V3 Calibration in closed loop | Are the probabilities trustworthy while the policy is acting? | Reliability of `p_arrive` at every STOP; `value` at t = 0 and per step; false-stop rate | ECE of `p_arrive` at STOP ≤ 0.05 (target) |
| V4 Cascade | How much System 2 is needed for how much gain? | Sweep thresholds; plot SR against % decisions escalated, with cost and wall-clock | Curve plus operating points at 5%, 10% and 20% escalation |
| V5 Robustness | How much does performance drop under shifts? | RxR-CE English val-unseen; HFOV 79° and 100°; camera height ±0.2 m; sliding off; image corruptions (blur, JPEG quality 30, brightness ±30%) | Report the SR drop per perturbation. No pass/fail in v1. |
| V6 Failure analysis | Does the model show the failure modes from the GPT-6-Astra report? | Classify all failures: false stop (STOP issued outside 3 m), passed-then-left (OSR = 1, SR = 0), route deviation (nDTW < 0.5), stuck (budget exhausted), never entered radius | Counts and example videos for each class |

### 11.2 Statistics

- Store a record per episode (outcome, metrics, number of escalations, latency) so all comparisons can be paired.
- Success rate: Wilson 95% interval. On the full val-unseen set this is about ±2.2 percentage points near SR 0.6. On R2R-CE-100 it is about ±8 points near SR 0.79, so R2R-CE-100 results are reported but never used alone to rank methods.
- Differences between two policies on the same episodes:
  - Exact McNemar test for SR.
  - Paired bootstrap (10,000 resamples) for SPL, nDTW and NE.
- ECE with a bootstrap interval.
- Evaluation uses argmax decisions (deterministic). Training-seed variance is reported from the 3 seeds for 9B.

### 11.3 Leakage and integrity checks

- envserver starts in either `train` or `eval` mode. In `eval` mode, privileged endpoints (oracle action, geodesic distance, position) return an error. Every eval run asserts eval mode in its manifest.
- The policy process has no file access to scene meshes or navmeshes.
- The scene-disjointness test from 6.1 runs on each dataset build.
- Pretraining exposure: Qwen's training data are not public, so MP3D imagery may have been seen during pretraining. Reports state this, as the GPT-6-Astra report does for its model.

## 12. Milestones and gates

The gate thresholds below are proposals. Confirm or change them before S1 starts, then do not move them after results come in.

| Milestone | Work | Gate |
|---|---|---|
| M0 Environment | envserver on the Mac (osx-arm64) with R2R-CE v1-3 and MP3D. Time-boxed attempt to build habitat-sim on the Spark. Measure rendering throughput and shard size. Check the Gated DeltaNet kernels on the Spark. Oracle nDTW audit (5.6). | **G0**: V0 passes |
| M1 Model plumbing | `build_stream`, `<act>` token, heads, hybrid-cache incremental inference, equivalence test, policyd API. Measure 9B and 27B training tokens/s and per-step latency. | **G1**: equivalence test passes; S0 overfit test passes |
| M2 Behaviour cloning | Feature extraction and S1f on 9B (B7, B8); head design fixed on cached features; then S1 and S1b; V1, V1b and V2 | **G2**: 9B SR on val-unseen ≥ 40% |
| M3 DAgger and calibration | S2 and S3 on 9B; V1–V3; data-scaling curve (7.4); ablations A1–A3, A6, A8 | **G3**: 9B SR ≥ 55% and `p_arrive` ECE at STOP ≤ 0.05 |
| M4 27B | Repeat S1–S3 on Qwen3.8-27B with the best 9B configuration | **G4**: 27B SR ≥ 9B SR + 3 points (McNemar p < 0.05), or report that the larger model does not help |
| M5 Cascade | B2, then V4 with System 2 = Qwen3.8-27B thinking; optional GPT-6-Astra; optional plan scorer (9.4) | **G5**: SR gain ≥ 5 points over s1nav alone at ≤ 20% escalated decisions |
| M6 Robustness and report | V5, V6, final tables and figures | None |

Ablations:
- A1: frame resolution 256 / 384 / 512.
- A2: STOP inside `move` versus the separate `arrive` rule.
- A3: one readout token versus one per head.
- A4: vision tower frozen versus last 4 blocks unfrozen.
- A5: with and without the noisy-oracle BC data.
- A6: hard negatives as a short refinement (S1b), versus mixed in from the start of S1, versus none. Measured on the probe set and the closed-loop false-stop rate.
- A7 (optional): instruction paraphrase augmentation, generated offline by the local Qwen model, following CoVer-VLA's use of rephrased instructions.
- A8: head trunk shared versus separate per head, and 3-layer versus 2-layer.

## 13. Compute budget (estimates, to be replaced by M1 measurements)

Assumptions:
- 30 decisions per training episode.
- About 100 tokens of header and instruction.
- Each step costs visual tokens + 6 tokens.
- Training costs about 6 × parameters FLOPs per token (LoRA with activation recomputation).
- Effective throughput 35 TFLOP/s: about 35% of the measured 99.8 TFLOPs BF16.
- One BC epoch covers the 10,819 R2R-CE training episodes.

| Frame | Tokens per episode | Tokens per epoch | 9B hours per epoch | 27B hours per epoch | Per-step latency 9B / 27B (BF16) |
|---|---|---|---|---|---|
| 256×256 | ≈2,200 | ≈24 M | ≈11 | ≈32 | ≈0.08 s / ≈0.23 s |
| 384×384 | ≈4,600 | ≈50 M | ≈23 | ≈66 | ≈0.09 s / ≈0.26 s |
| 512×512 | ≈8,000 | ≈86 M | ≈40 | ≈114 | ≈0.12 s / ≈0.36 s |

Implications at 384×384:
- 9B: S1 plus four DAgger rounds is on the order of one to two weeks of Spark time, including rollouts.
- 27B: about three times that.
- One full val-unseen evaluation: about 2 h for 9B and 5 h for 27B of model time if run one episode at a time. With batching (13.1) it drops to about 1.2 h and 3.4 h.

These figures are why the 27B run is gated on 9B results, and why A1 (lower resolution) matters.

### 13.1 Closed-loop throughput and batching

- At 384×384, each decision adds about 150 tokens. That costs about 2.9 TFLOP (9B) or 8.3 TFLOP (27B) per decision, so the model is compute-bound even at batch size 1.
- Batching across episodes therefore helps only modestly: it hides the weight-read time and per-call overhead. Expected ceiling at 50 TFLOP/s effective:
  - 9B: about 17 decisions/s.
  - 27B: about 6 decisions/s.
- **Requirement:** policyd accepts a batch of `decide` calls from several concurrent episodes (default 8) and runs them in one forward pass with per-episode caches. envserver runs one process per concurrent episode.

| Workload | Decisions | 9B hours | 27B hours |
|---|---|---|---|
| One DAgger round (5,000 episodes × ~40 decisions) | ~200k | ~3.2 | ~9.3 |
| One full val-unseen evaluation (1,839 × ~40) | ~74k | ~1.2 | ~3.4 |

### 13.2 Simulator budget (Habitat)

The simulator is a small share of total compute. The model on the Spark sets the pace.

| Item | Frames |
|---|---|
| BC rollouts (oracle and noisy oracle, 10,819 episodes × 30 × 2) | ~650k |
| DAgger, 4 rounds | ~800k |
| Extra on-policy value episodes | ~80k |
| About 20 full val-unseen evaluations over the project | ~1.5M |
| **Total** | **~3.0M** |

- **Rendering time.** Habitat's documentation reports several thousand frames/s single-threaded on MP3D, on a 2016-era GPU at small resolution. Throughput on a Mac at 384×384 is unknown. At a conservative 100 frames/s the whole project needs about 8 h of rendering; at 500 frames/s about 2 h. BC rendering has no model in the loop and can run overnight on the Mac.
- **In closed loop the simulator waits on the model.** 17 decisions/s is far below what one Habitat process can render, so 4–8 envserver processes on the Mac are enough. Each holds one MP3D scene in memory; M0 measures per-process memory.
- **Network.** About 30 KB per JPEG frame × ~17 frames/s ≈ 0.5 MB/s. Any wired LAN is sufficient.
- **Storage.** About 45 GB of training frames at 30 KB each, plus the MP3D scene download and R2R-CE episode files.
- **M0 benchmark.** Measure frames/s for 1, 4 and 8 processes at 256/384/512, peak memory per process, and JPEG size. Record the numbers in the manifest and replace this table's estimates.

### 13.3 Frozen-feature stage budget

| Item | 9B | 27B |
|---|---|---|
| One extraction pass over BC data (~50M tokens at 384×384, 2 × params FLOPs/token, 50 TFLOP/s) | ~5 h | ~15 h |
| Feature storage, BC data (~650k decisions × 8 or 10 KB) | ~5 GB | ~7 GB |
| One S1f head training run | minutes (to be measured) | minutes (to be measured) |

The CLM authors report that a full head pre-training run on 60M pairs takes about 1 hour on one RTX 4090, once embeddings are cached.

## 14. Risks and open decisions

| Risk | Effect | Mitigation |
|---|---|---|
| No aarch64 habitat-sim package | Simulator cannot run on the Spark | Mac envserver over LAN (default); source build (time-boxed); x86 fallback |
| Gated DeltaNet kernels unavailable or slow on the Spark | Training much slower than estimated | M1 measures; fall back to 9B only, or rent x86 GPUs for 27B |
| Hybrid cache API does not support clean incremental forward | Latency advantage lost | Equivalence test at M1; fallback is windowed recomputation (slower, still no text generation) |
| Different renderers across hosts | Results not comparable | V0 on each host; renderer string in manifest |
| MP3D access | No scenes | User signs the Matterport Terms of Use early |
| Calibration degrades on unseen scenes | Cascade triggers wrongly | V3 measures on val-unseen, not only on the calibration split; thresholds from the calibration split only |
| Teacher terms of use | S4 or B6 not permitted | Keep S4 optional; System 2 default is the local Qwen model |
| 27B gives no gain over 9B | Wasted weeks | Gate G4; report a null result |
| MP3D region and object annotations missing or unreliable for some scenes | Landmark near-misses (6.3, type 2) cannot be mined | Check at M0; fall back to distance and passed-then-left near-misses only |
| Frozen features too weak for navigation | S1f results do not predict LoRA results | Confirm the best S1f choice once with LoRA before relying on S1f for further decisions |

Open decisions for the user:

1. Confirm the gate thresholds in Section 12.
2. Host placement (Mac envserver or Spark source build).
3. Whether GPT-6-Astra is used at all (B6, S4, or the System 2 option).
4. Whether the 27B model is the deployment target or the 9B model is enough.

## 15. Implementation notes for Claude Code

### 15.1 Repository layout

```
s1nav/
  protocol/        shared RPC schema (dataclasses + msgpack), no heavy imports; Python 3.9-compatible
  envserver/       habitat wrapper, macro expansion, oracles, metrics (SR, SPL, nDTW, NE, OSR),
                   train/eval mode guard, episode logging, MCP adapter for LLM agents
  data/            rollout jobs, shard writer, dataset manifest, scene-disjointness check
  model/           backbone loader, <act> token, heads, build_stream(), hybrid-cache stepping
  train/           S0–S3 loops, DAgger orchestration, calibration fitting
  policyd/         inference server, escalation controller, System 2 client
  eval/            runners V0–V6, statistics, reliability plots, report builder
  agents/          baselines B0–B2
  configs/         one YAML per experiment; every run writes a resolved copy
  tests/
  envs/            envserver (conda, py3.9) and policy (py3.12) environment files, both pinned
```

### 15.2 Required tests

1. Incremental versus full-sequence head equivalence (5.3).
2. `build_stream` gives identical output in trainer and policyd, including at the window rebuild.
3. Macro expansion and oracle-to-macro round trip (5.5).
4. Metrics match habitat-lab measures on recorded trajectories; SPL and nDTW match hand-computed toy cases.
5. Eval-mode guard rejects privileged calls.
6. Input builder reads only whitelisted fields.
7. Scene disjointness between training/calibration data and val-unseen.
8. `arrive` and `progress` labels match geodesic distances recomputed from logged positions.
9. Determinism: same checkpoint, same episode and same config give the same trajectory.
10. ECE and Brier implementations against a small known example.
11. Feature-cache equivalence: for 20 logged episodes, features stored by the S1f extractor match the hidden states policyd computes live with the adapter off (max absolute difference below 1e-2 in BF16).
12. Probe-set integrity: the probe set contains only calibration-scene states, its hash is fixed after M1, and no training job reads it.
13. Mined-negative labels: every type 1–3 state has geodesic distance ≥ 3.0 m, and every matched positive < 3.0 m, recomputed from logged positions.

### 15.3 Run manifest (written by every train and eval job)

- Git commit and resolved config.
- Host, OS, GPU and GL renderer strings.
- habitat-sim and habitat-lab versions.
- Dataset version and shard hashes; scene list hash.
- Model id and commit hash; LoRA module list; checkpoint hash.
- Seeds.
- Temperatures and thresholds.
- envserver mode.
- `abort_on_block` and sliding settings.

### 15.4 Build order

1. `protocol/` and a minimal `envserver/` with B0/B1 and metrics (G0).
2. Data rollouts and shards.
3. `model/` with `build_stream` and the equivalence test (G1).
4. S0, feature extraction and S1f, then S1, S1b and eval runners (G2).
5. DAgger and calibration (G3).
6. 27B (G4).
7. System 2, cascade and B2 (G5).
8. Robustness and report.

## 16. Out of scope for v1

- Real-robot deployment and sim-to-real.
- Runtime-defined questions in the Jev sense (arbitrary question text at inference). The fixed heads cover navigation; a general question interface would need a question-diverse training set.
- Reinforcement-learning fine-tuning on task reward. This is a candidate for v2 once DAgger results are known.
- Depth, panoramas and prebuilt maps (monocular RGB only, as in the GPT-6-Astra report).
- Other simulators (user decision, 26 September 2026: Habitat only). Isaac Sim / VLN-PE, Cosmos-based restyling and Image2Sim data were considered and deferred. Habitat R2R-CE is the only setting whose results compare directly with the literature.
- A learned world model. This design can host one later: a latent predictor could supply imagined next states for the `arrive` head to score.
- Separate state and action encoders with action-embedding caching (CLM's design). With 10 fixed macros read in one pass, there is nothing to cache.
