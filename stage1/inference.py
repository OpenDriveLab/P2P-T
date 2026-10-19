import argparse
import contextlib
import os
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from stage1.config_utils import (
    build_stage1_model_kwargs,
    load_stage1_config,
    resolve_torch_dtype,
    resolve_stage1_checkpoint_path,
    supports_autocast,
)
from stage1.dataset import apply_delta_pose
from stage1.model.stage1_transformer import Stage1Transformer


def preprocess_tool_pose(current_tool_pose: Iterable[float]) -> np.ndarray:
    pose = np.asarray(current_tool_pose, dtype=np.float32).squeeze()
    if pose.shape == (4, 4):
        return pose.reshape(-1)
    if pose.shape == (16,):
        return pose
    raise ValueError(
        f"Expected current_tool_pose to be a 4x4 matrix or flattened 16-vector, got shape {pose.shape}"
    )


def postprocess_prediction(
    prediction: np.ndarray,
    current_tool_pose: np.ndarray,
    prediction_target: str,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim == 1:
        prediction = prediction[None, :]
    if prediction.shape[-1] != 16:
        prediction = prediction.reshape(-1, 16)
    if prediction_target == "delta_pose":
        current_pose_matrix = current_tool_pose.reshape(4, 4)
        prediction = np.stack(
            [
                apply_delta_pose(current_pose_matrix, delta_pose.reshape(4, 4))
                for delta_pose in prediction
            ],
            axis=0,
        )
    return prediction.reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 inference.")
    parser.add_argument(
        "--config",
        default="stage1/config/stage1.yaml",
        help="Path to a Stage 1 YAML config.",
    )
    return parser.parse_args()


def build_model_from_config(config: dict, checkpoint_path: str, device: torch.device):
    model = Stage1Transformer.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
        **build_stage1_model_kwargs(config),
    )
    inference_dtype = resolve_torch_dtype(config.get("inference", {}).get("model_dtype"))
    model.eval()
    if inference_dtype is not None and device.type == "cuda":
        model.to(device=device, dtype=inference_dtype)
    else:
        model.to(device)
    return model


def infer(
    image_path: str,
    instruction: str,
    current_tool_pose: Iterable[float],
    config_path: str = "stage1/config/stage1.yaml",
) -> np.ndarray:
    config = load_stage1_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_stage1_checkpoint_path(
        checkpoint_path=config.get("inference", {}).get("checkpoint_path"),
        checkpoint_dir=config.get("inference", {}).get("checkpoint_dir"),
        checkpoint_name=config.get("inference", {}).get("checkpoint_name", "last.ckpt"),
    )

    model = build_model_from_config(config, str(checkpoint_path), device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    current_pose = preprocess_tool_pose(current_tool_pose)
    pose_tensor = torch.from_numpy(current_pose).unsqueeze(0).to(device)
    autocast_dtype = resolve_torch_dtype(config.get("inference", {}).get("autocast_dtype"))
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if supports_autocast(device, autocast_dtype)
        else contextlib.nullcontext()
    )

    with torch.no_grad(), autocast_context:
        predicted_pose = model(image_tensor, pose_tensor, [instruction])

    prediction_target = config.get("data", {}).get("prediction_target", "next_pose")
    predicted_pose_np = predicted_pose.cpu().numpy()
    processed_prediction = postprocess_prediction(
        predicted_pose_np[0],
        current_pose,
        prediction_target,
    )
    predicted_pose_np[0] = processed_prediction.reshape(predicted_pose_np[0].shape)
    return predicted_pose_np


if __name__ == "__main__":
    args = parse_args()
    sample_image = "data/stage1_data/parsed_taco_data/dummy_task/seq_0/rgb/000000.png"
    sample_pose = np.eye(4, dtype=np.float32)
    if os.path.exists(sample_image):
        print(infer(sample_image, "pick up the tool", sample_pose, config_path=args.config))
    else:
        print(f"Sample image not found: {sample_image}")
