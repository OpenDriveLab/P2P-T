#!/usr/bin/env bash
set -euo pipefail

# P2P-T Stage 2: freeze the Stage 1 object-centric world model and fine-tune
# the RDT action expert plus its pose-latent adaptor.

CONFIG_PATH="${CONFIG_PATH:-configs/ur10e_p2p_t.yaml}"
STAGE1_CONFIG_PATH="${STAGE1_CONFIG_PATH:-stage1/config/stage1.yaml}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-}"
RDT_PRETRAINED_MODEL="${RDT_PRETRAINED_MODEL:-robotics-diffusion-transformer/rdt-1b}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-google/t5-v1_1-xxl}"
VISION_ENCODER_NAME="${VISION_ENCODER_NAME:-google/siglip-so400m-patch14-384}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/p2p_t_stage2_ur10e}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing P2P-T config: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$STAGE1_CONFIG_PATH" ]]; then
  echo "Missing Stage 1 config: $STAGE1_CONFIG_PATH" >&2
  exit 1
fi
if [[ -z "$STAGE1_CHECKPOINT_PATH" ]]; then
  echo "Set STAGE1_CHECKPOINT_PATH to a released Stage 1 checkpoint." >&2
  exit 1
fi
if [[ ! -f "$STAGE1_CHECKPOINT_PATH" && ! -d "$STAGE1_CHECKPOINT_PATH" ]]; then
  echo "Stage 1 checkpoint does not exist: $STAGE1_CHECKPOINT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

accelerate launch main.py \
  --config_path="$CONFIG_PATH" \
  --pretrained_model_name_or_path="$RDT_PRETRAINED_MODEL" \
  --pretrained_text_encoder_name_or_path="$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path="$VISION_ENCODER_NAME" \
  --stage1_config_path="$STAGE1_CONFIG_PATH" \
  --stage1_checkpoint_path="$STAGE1_CHECKPOINT_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --dataset_type=p2p_t \
  --load_from_hdf5 \
  --use_stage1_condition \
  --mixed_precision=bf16 \
  --report_to=wandb \
  "$@"

