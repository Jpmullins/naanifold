# jev_vln

A fast, calibrated vision-and-language navigation model in the style of TypeSafe's Jev. It uses a Qwen vision-language backbone: Qwen3.5-9B for development and Qwen3.8-27B as the final model. It is trained and evaluated in Habitat on R2R-CE. Working model name: `s1nav`.

## Contents

| Path | What it is |
|---|---|
| `docs/s1nav_spec.md` | Build specification for Claude Code: architecture, data, training, calibration, evaluation, milestones and gates, compute budget |
| `docs/background_jev_vln_world_models.md` | How Jev works, what the GPT-6-Astra VLN report shows, the Jev-like VLN design, and how it differs from a world model |
| `docs/simulator_options.md` | Simulator decision (Habitat only), facts checked, options considered and deferred |
| `docs/clm8b_vs_jev.md` | CLM-8B (Stanford / NVIDIA Research): how it compares with Jev, first look at what s1nav can borrow |
| `docs/clm_build_review.md` | Code-level review of how CLM-8B was built and trained (repo, checkpoint, dataset cards, authors' blog), with the adopt/reject decisions applied in spec v0.2 |
| `references/gpt6_astra_vln_arXiv_2609.29861.pdf` | "GPT-6-Astra Lights Up Embodied Navigation", Dai et al., arXiv:2609.29861 |
| `artifacts/` | Outputs from the build: run manifests, evaluation tables, figures |

## Status (26 September 2026)

- Spec v0.2 written: adds a frozen-feature head stage, hard-negative refinement for arrival detection, a near-miss probe set and a pre-registered checkpoint metric, all taken from the CLM-8B review. Nothing built or run yet.
- Decided: Habitat only; develop on 9B and move to 27B only if 9B passes gate G3.

## Open items for the user

1. Request Matterport3D scene access (Matterport Terms of Use, non-commercial research).
2. Confirm or change the gate thresholds G0–G5 (spec Section 12) before training starts.
3. Decide whether GPT-6-Astra is used at all. Using its outputs as training labels requires a terms-of-use check.
4. Decide whether 27B or 9B is the deployment target.
