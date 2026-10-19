# 🎥 Stage 1 preprocessing contract

The P2P-T data path converts an RGB video and a language instruction into
validated, object-centric training samples. The repository keeps the
foundation-model runtimes separate because Grounded-SAM, SAM 3D, and
FoundationPose each have their own CUDA and license requirements.

For every `task/seq_*` directory, the release contract is:

```text
seq_000/
  rgb/000000.png ...
  masks/000000.png ...          # Grounded-SAM binary tool masks
  tool_mesh.obj                 # SAM 3D instance mesh
  tool_poses.pkl                # FoundationPose poses, shape (N, 4, 4)
  instruction.txt               # task language
  verification.json             # human accept/reject and discard reason
```

Run the validator before Stage 1 training:

```bash
python -m preprocess.validate_stage1_dataset data/stage1_data/parsed_taco_data \
  --future-steps 8
```

The pose matrices are camera-relative homogeneous transforms. Keep the RGB
frame numbering aligned with the pose index, preserve the original videos and
foundation-model versions in a manifest, and record clips rejected during the
human verification pass. The training dataset samples the current frame and
the next `future_steps` frames, so a sequence must contain at least
`future_steps + 1` aligned poses and images.

