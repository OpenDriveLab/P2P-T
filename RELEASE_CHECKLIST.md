# ✅ ToolUse-RDT release checklist

This checklist separates the paper's required inputs from repository release
artifacts. A checked box means the artifact has been produced, verified, and
recorded in the release manifest; it does not mean a future checkpoint is
already included in Git.

## Stage 1 - preprocess and dataset

- [ ] Preserve raw egocentric/allocentric videos and source licenses.
- [ ] Run Grounded-SAM and export one binary tool mask per RGB frame.
- [ ] Run SAM 3D and export the instance mesh (`tool_mesh.obj`).
- [ ] Run FoundationPose and export camera-relative `(N, 4, 4)` poses.
- [ ] Keep frame numbering, RGB images, masks, mesh, and poses aligned.
- [ ] Record language instruction, tool category, camera calibration, model
      versions, and preprocessing command in a manifest.
- [ ] Complete human verification; record accepted/rejected clips and discard
      reasons. The paper reports a 27% overall discard rate.
- [ ] Run `python3 -m preprocess.validate_stage1_dataset ... --future-steps 8`.
- [ ] Publish dataset counts and the train/validation split seed.

## Stage 1 - object-centric world model

- [ ] Check in the resolved YAML config and encoder revisions (DINOv2,
      SigLIP, T5-base).
- [ ] Confirm `future_steps` and pose representation (flattened 4x4 matrices).
- [ ] Train with `L_pose + 0.01 * L_frame`; inspect both metrics.
- [ ] Save the best and last Stage 1 checkpoints with optimizer, scheduler,
      config, and dataset manifest hashes.
- [ ] Register the Stage 1 pretrained checkpoint URI, SHA256, and expected
      `future_steps` in the release manifest.
- [ ] Run offline inference on held-out videos and inspect pose overlays.

## Stage 2 - pose-aware post-training

- [ ] Record the base RDT checkpoint URI and revision.
- [ ] Load the frozen Stage 1 checkpoint and verify the future latent token
      count matches the Stage 2 config.
- [ ] Fine-tune only the RDT action expert and pose adaptor; keep Stage 1,
      T5, and SigLIP frozen.
- [ ] Save post-trained checkpoints with the Stage 1 and base-RDT references,
      robot embodiment, action chunk size, and dataset statistics.
- [ ] Check that the Stage 2 checkpoint contains the expanded RDT positional
      embedding for `1 + future_steps` state tokens.
- [ ] Run the deployment smoke test, including receding-horizon replanning,
      camera ordering, action dimensions, and control frequency.
- [ ] Report per-task success, inference rate, and the six paper tasks where
      applicable (hammer, cup, brush, screwdriver, knife, spoon).

## Release gate

- [ ] `python3 scripts/validate_release.py --config configs/ur10e_p2p_t.yaml`
      passes without errors.
- [ ] `python3 -m compileall stage1 models preprocess scripts train` passes.
- [ ] No credentials, raw videos, generated logs, or checkpoints are tracked.
- [ ] `README.md`, configs, `release_manifest.yaml`, and known limitations are
      reviewed together as one versioned release.
- [ ] `train_stage2.sh` runs from a clean checkout with only external dataset
      and checkpoint paths supplied through environment variables.

## Future artifact register

| Artifact | Required metadata | Status |
| --- | --- | --- |
| Stage 1 preprocess output | dataset root, manifest, model revisions, hashes | Pending |
| Stage 1 pretrained checkpoint | URI, SHA256, config, `future_steps`, metrics | Pending |
| Stage 2 post-trained checkpoint | URI, SHA256, base RDT, Stage 1 URI, robot/config | Pending |
| Deployment calibration | camera intrinsics/extrinsics, tool pose convention | Pending |

The editable placeholder is [`release_manifest.yaml`](release_manifest.yaml).

