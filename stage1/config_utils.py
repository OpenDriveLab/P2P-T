from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import yaml


DEFAULT_STAGE1_CONFIG_PATH = Path("stage1/config/stage1.yaml")


def load_stage1_config(config_path: Optional[str] = None) -> dict:
    path = Path(config_path) if config_path is not None else DEFAULT_STAGE1_CONFIG_PATH
    if not path.is_absolute() and not path.is_file():
        path = Path(__file__).resolve().parent.parent / path
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_stage1_checkpoint_path(
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_name: str = "last.ckpt",
) -> Path:
    candidates: list[Path] = []
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))
    if checkpoint_dir:
        candidates.append(Path(checkpoint_dir))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            matches = sorted(
                candidate.rglob(checkpoint_name),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return matches[0]

    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        f"Unable to find Stage 1 checkpoint '{checkpoint_name}' under: {searched}"
    )


def build_stage1_model_kwargs(config: dict) -> dict[str, Any]:
    data_config = config.get("data", {})
    configured_future_steps = data_config.get(
        "future_steps", data_config.get("recipe", {}).get("future_steps", 8)
    )
    return {
        "embed_dim": config["model"]["embed_dim"],
        "num_heads": config["model"]["num_heads"],
        "num_layers": config["model"]["num_layers"],
        "feedforward_mult": config["model"].get("feedforward_mult", 8),
        "num_fusion_tokens": config["model"].get("num_fusion_tokens", 32),
        "image_feature_dim": config["model"]["image_feature_dim"],
        "language_feature_dim": config["model"]["language_feature_dim"],
        "pose_dim": config["model"]["pose_dim"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"].get("weight_decay", 1e-4),
        "lr_scheduler": config["training"].get("lr_scheduler", "constant"),
        "lr_warmup_steps": config["training"].get("lr_warmup_steps", 0),
        "dino_repo": config["model"].get("dino_repo", "facebookresearch/dinov2"),
        "dino_model_name": config["model"].get("dino_model_name", "dinov2_vitb14_reg"),
        "siglip_model_name": config["model"].get("siglip_model_name", "google/siglip-base-patch16-224"),
        "text_model_name": config["model"].get("text_model_name", "t5-base"),
        "freeze_backbones": config["model"].get("freeze_backbones", True),
        "cache_language_embeddings": config["model"].get("cache_language_embeddings", True),
        "max_text_length": config["model"].get("max_text_length", 512),
        "future_steps": config["model"].get("future_steps", configured_future_steps),
        "frame_loss_weight": config["model"].get("frame_loss_weight", 0.01),
    }


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None

    normalized = str(dtype_name).strip().lower()
    if normalized in {"", "none", "null", "auto"}:
        return None

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of: {valid}.") from exc


def supports_autocast(device: torch.device, dtype: Optional[torch.dtype]) -> bool:
    return device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
