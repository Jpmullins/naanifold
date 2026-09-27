# Background: Jev, a Jev-like VLN model, and world models

Notes from the 26 September 2026 design discussion. Sources were checked that day.

## 1. Jev (TypeSafe AI)

### Public facts

- Released in early access on 15 September 2026. TypeSafe calls it a "System One model".
- Returns typed decisions with probabilities, not text. There are three question types, all defined at runtime: yes/no, choice and score. All questions are answered in one parallel pass.
- Input is text only.
- The stack has three named parts: a new architecture, a parallel sampler, and a training method called RLCD (Reinforcement Learning for Calibrated Decisions).
- It is described as transformer-based and trained only on synthetic data. The architecture and base model are not disclosed; outside observers suspect an open-weight LLM underneath.
- Schema validity comes from the architecture: the answer space is fixed in advance. RLCD only governs calibration. TypeSafe does not claim RLCD improves accuracy.
- TypeSafe's "workflow evals" score models against the average of GPT-6 Astra and Claude Fable 5.1. They therefore measure agreement with frontier models, not accuracy against ground truth. On that measure Jev scores 67.8%, level with GPT-5.6 Terra (67.9%) and behind Opus 5 (73.1%), at much lower cost and latency.
- The approach is easy to approximate. Together AI fine-tuned a Jev-like classifier on Qwen3.5 4B from about 38k examples, for about $17.

### Interpretation (ours, not TypeSafe's)

- Jev looks like frontier-model judgement distilled into a fast classifier, with a calibration objective added.
- If the reward is a proper scoring rule against a fixed label, the RL framing gives the same result as ordinary supervised cross-entropy training. RL adds something only when outcomes are random or delayed.
- Calibration at one step does not automatically carry over to a whole sequence of steps. This matters for navigation, where an episode has tens to hundreds of decisions.

Sources:
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://en.wikipedia.org/wiki/Jev_(AI_model)
- https://www.turingpost.com/p/what-is-jev-rlcd
- https://www.sanity.io/glossary/rlcd-reinforcement-learning-for-calibrated-decisions
- https://benchlm.ai/blog/posts/what-is-jev
- https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e
- https://www.together.ai/blog/how-to-train-your-own-jev

## 2. The GPT-6-Astra VLN report (references/gpt6_astra_vln_arXiv_2609.29861.pdf)

**Setup.**
- Zero-shot R2R-CE, monocular 512×512 RGB, driven through the Codex harness.
- The model sees the scene only through two tools: `observe()`, which returns the current view, and `step(actions)`, which executes primitive actions.
- Primitives: forward 0.25 m, turn 15°, tilt 30°, STOP.

**Results.**

| Reasoning setting | Success rate (SR) | SPL |
|---|---|---|
| Medium | 76.0% | 67.9% |
| Ultra | 79.0% | 69.5% |

- Evaluation is on the 100-episode Open-Nav subset, one run each. Our Wilson 95% interval for 79/100 is about 70–86%.
- Medium versus ultra: 9 episodes flipped one way and 6 the other. The two runs also used different Codex versions, so the difference between settings is within noise.

**Failure modes.** Route deviation despite plausible landmark matches, not noticing that motion had no effect, and false arrival claims. The model called STOP in 99 of 100 episodes. Three failed episodes had passed within the success radius before stopping elsewhere.

**Caveats stated by the authors.**
- Comparisons with trained methods use a different split.
- The training data are unknown, so "zero-shot" means no navigation fine-tuning. It does not rule out prior exposure to navigation data.

## 3. A Jev-like model for VLN

A single forward pass per step answers a fixed set of questions:
- which move to make;
- am I at the goal;
- how far along the path am I;
- did my last move fail;
- will this episode succeed.

The probabilities are calibrated. When they show uncertainty, the controller hands control to a slower reasoning model.

Differences from Jev:
- It needs vision input.
- Labels can come from the simulator (correct move, true distance to goal), not from a teacher model, so calibration is against real outcomes.
- An episode-level success head is needed, because step-level calibration does not compose across an episode.

Expected performance. These are reasoning, not measurements:

| Design | Expected success rate |
|---|---|
| Jev as-is with captioned frames | Below ~40% |
| Vision student trained on simulator labels plus correction rounds | ~60–70% on full val-unseen, similar to existing trained monocular models |
| With escalation to a reasoning model | Could approach the teacher, at a fraction of the reasoning-model calls |

The main gain is latency: a fraction of a second per step, against up to 2,400 s per episode for GPT-6-Astra.

The full design is in `s1nav_spec.md`.

## 4. Jev-like model versus a world model (JEPA sense)

| | Jev-like policy | JEPA-style world model |
|---|---|---|
| Learns | p(action \| observation, instruction); p(arrived \| observation, instruction) | p(next latent state \| latent state, action) |
| Answers | What should I do, and is this true now? | What happens if I do this? |
| Training signal | Labelled decisions (simulator, teacher, outcomes) | Observation–action–next-observation triples, self-supervised |
| Chooses actions alone | Yes | No; it needs a planner and a cost function |
| Counterfactuals and look-ahead | No | Yes |

The two combine naturally. The world model imagines where candidate action sequences end up, and the Jev-like "arrived?" head scores each imagined end state, acting as the planner's cost function.

Related work:
- Navigation World Models (Bar et al., CVPR 2025).
- V-JEPA 2-AC (planning for manipulation).
- AstraNav-World (67.9% SR, panoramic, in the Astra report's Table 1).
