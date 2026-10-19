# 🧩 P2P-T implementation map

The attached P2P-T paper maps to this repository as follows:

| Paper component | Repository implementation |
| --- | --- |
| Grounded-SAM, SAM 3D, FoundationPose | `preprocess/` output contract; external model runners are intentionally not bundled |
| Flattened 4x4 tool pose | `stage1/dataset.py` and `stage1/inference.py` |
| Frozen DINOv2 + SigLIP + T5-base | `stage1/model/stage1_transformer.py` |
| Future pose and visual latent prediction | `Stage1Transformer.future_steps`, `pose_latents`, and `frame_head` |
| `L_pose + 0.01 L_frame` | `Stage1Transformer._shared_step` |
| Frozen Stage 1 model in Stage 2 | `models/stage1_conditioned_rdt_runner.py` |
| Pose-latent adaptor before proprioception | `target_pose_adaptor` and expanded `num_state_tokens` |
| RDT diffusion action expert | `models/rdt_runner.py` and `models/rdt/model.py` |

The canonical example configuration is
`configs/ur10e_p2p_t.yaml`. It enables Stage 1 conditioning, uses eight future
pose tokens, and keeps the paper's 64-step action chunk. Paths to data and
weights remain local configuration values; large binary artifacts belong in a
model or dataset registry and are recorded in `release_manifest.yaml` and
`RELEASE_CHECKLIST.md`.

