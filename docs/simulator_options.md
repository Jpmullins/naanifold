# Simulator options and decision

**Decision (user, 26 September 2026): Habitat only.** R2R-CE is defined in Habitat on Matterport3D scenes. Every result we compare against, including GPT-6-Astra's 79% and the trained models in its Table 1, comes from that setting.

## Habitat facts checked on 26 September 2026

- habitat-sim 0.3.3 conda builds exist for linux-64 and osx-arm64 only. There is no linux-aarch64 build, so the DGX Spark has no prebuilt package. Python 3.9 only.
- Headless rendering relies on EGL, which does not work on macOS. The Mac needs a windowed GL context.
- habitat-lab main includes the `VLN-v0` task, the `R2RVLN-v1` dataset loader, `ShortestPathFollower` and `benchmark/nav/vln_r2r.yaml`.
- The VLN-CE reference repo (MIT licence) is pinned to habitat-sim 0.1.7. Its reference values: success radius 3.0 m, `ALLOW_SLIDING: True`, RGB 224×224, HFOV 90.
- Habitat's documentation reports several thousand frames/s single-threaded on MP3D, and over 10,000 frames/s multi-process on one GPU. This is a 2016-era GPU at small resolution. Mac throughput is to be measured at M0.

## Placement

1. **Default.** envserver on the Mac; the model on the Spark; connected over LAN.
2. **Time-boxed attempt.** Build habitat-sim from source on the Spark.
3. **Fallback.** A rented x86 Linux GPU host.

## Compute

The simulator is a small share of total compute: about 3M frames over the whole project, which is hours of rendering. The model on the Spark sets the pace, at about 17 decisions/s for 9B and 6/s for 27B (estimated). Four to eight envserver processes are enough. Details are in `s1nav_spec.md`, Section 13.

## Considered and deferred

| Option | What it is | Why deferred |
|---|---|---|
| Isaac Sim + VLN-PE (InternNav / GRUtopia) | Physically realistic VLN: same 90 MP3D scenes converted to USD, VLN-CE episode poses ported, humanoid/quadruped/wheeled controllers. Runs on the DGX Spark, but needs DISPLAY set even with --headless. | Separate leaderboard, not comparable with R2R-CE. Candidate for a later cross-simulator test. |
| VLNVerse (Isaac Sim) | 263 hand-crafted physics-aware scenes | Same comparability issue |
| NVIDIA Cosmos (Predict2.5, Transfer2.5, Policy) | Generative video world models | No ground-truth floor plan: no geodesic distance, no success scoring, no oracle labels. Possible later uses: Transfer2.5 restyling of Habitat frames for visual variety, or as a world-model planner. |
| Image2Sim (arXiv:2607.05765) | Gaussian-anchored neural simulator; ~20K scenes, >10M samples | Training-data source, not a benchmark; generated data are CC BY-NC 4.0 |

Sources:
- https://github.com/facebookresearch/habitat-sim
- https://github.com/facebookresearch/habitat-lab
- https://github.com/jacobkrantz/VLN-CE
- https://arxiv.org/pdf/2507.13019 (VLN-PE)
- https://github.com/InternRobotics/InternNav
- https://arxiv.org/abs/2512.19021 (VLNVerse)
- https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_isaac_robotics/2_isaac_installation/
- https://arxiv.org/abs/2511.00062 (Cosmos-Predict2.5 / Transfer2.5)
- https://huggingface.co/blog/nvidia/cosmos-policy-for-robot-control
- https://arxiv.org/abs/2607.05765 (Image2Sim)
- https://github.com/MrZihan/Image2Sim
