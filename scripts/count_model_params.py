#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CountRow:
    name: str
    total: int
    trainable: Optional[int] = None
    frozen: Optional[int] = None
    source: str = ""
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count whole-model parameters for the Stage 1 conditioned Stage 2 "
            "model from this repository's configs/checkpoints."
        )
    )
    parser.add_argument("--stage1-config-path", default="stage1/config/stage1.yaml")
    parser.add_argument("--stage2-config-path", default="configs/ur10e_p2p_t.yaml")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument(
        "--stage1-count-source",
        choices=("auto", "checkpoint", "load", "formula"),
        default="auto",
        help=(
            "auto: use a Stage 1 checkpoint if found, otherwise report the "
            "non-backbone formula count. checkpoint: count tensors in a local "
            "checkpoint. load: instantiate Stage1Transformer, including external "
            "backbones. formula: count only Stage 1 non-backbone modules."
        ),
    )
    parser.add_argument(
        "--allow-stage1-model-load",
        action="store_true",
        help=(
            "In auto mode, instantiate Stage1Transformer if no checkpoint is "
            "found. This may load/download DINOv2, SigLIP, and T5 weights."
        ),
    )
    parser.add_argument("--stage1-checkpoint-path", default=None)
    parser.add_argument("--stage1-checkpoint-dir", default=None)
    parser.add_argument("--stage1-checkpoint-name", default=None)
    parser.add_argument(
        "--stage2-checkpoint-path",
        default=None,
        help=(
            "Optional local Stage 2 checkpoint/model file or directory to count "
            "directly. If it contains embedded Stage 1 tensors, the checkpoint "
            "count is already the whole model count."
        ),
    )
    parser.add_argument(
        "--use-stage1-condition",
        action="store_true",
        help=(
            "Count Stage 2 as the Stage1ConditionedRDTRunner shape. This is also "
            "enabled when stage1.enabled is true in the Stage 2 config."
        ),
    )
    parser.add_argument(
        "--vision-num-patches",
        type=int,
        default=729,
        help=(
            "Number of patch tokens produced by the Stage 2 vision encoder. "
            "google/siglip-so400m-patch14-384 uses 729."
        ),
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Also print the intermediate Stage 1 and Stage 2 component rows.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_count(value: Optional[int]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1_000_000_000:
        return f"{value:,} ({value / 1_000_000_000:.3f}B)"
    if abs(value) >= 1_000_000:
        return f"{value:,} ({value / 1_000_000:.3f}M)"
    return f"{value:,}"


def resolve_checkpoint_path(
    checkpoint_path: Optional[str],
    checkpoint_dir: Optional[str],
    checkpoint_name: str,
) -> Optional[Path]:
    candidates: list[Path] = []
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))
    if checkpoint_dir:
        candidates.append(Path(checkpoint_dir))

    for candidate in candidates:
        candidate = candidate.expanduser()
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
    return None


def resolve_model_file(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    for filename in ("model.safetensors", "pytorch_model.bin", "last.ckpt"):
        candidate = path / filename
        if candidate.is_file():
            return candidate
    matches = sorted(path.rglob("mp_rank_00_model_states.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find a known model file under: {path}")


def load_tensor_dict(path: Path) -> tuple[dict[str, Any], str]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except ImportError as exc:
            raise ImportError("Install safetensors to read .safetensors files.") from exc
        return load_safetensors_file(str(path), device="cpu"), "safetensors"

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("state_dict", "module", "model"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value, key
        return obj, "top-level"
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")


def count_state_tensors(
    state: dict[str, Any],
    *,
    include_prefix: Optional[str] = None,
    exclude_prefix: Optional[str] = None,
) -> int:
    total = 0
    for key, value in state.items():
        if include_prefix is not None and not key.startswith(include_prefix):
            continue
        if exclude_prefix is not None and key.startswith(exclude_prefix):
            continue
        if torch.is_tensor(value):
            total += value.numel()
    return total


def module_count_row(name: str, module: torch.nn.Module, source: str) -> CountRow:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return CountRow(
        name=name,
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        source=source,
    )


def linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def adapter_params(projector_type: str, in_features: int, out_features: int) -> int:
    if projector_type == "linear":
        return linear_params(in_features, out_features)

    match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if match is None:
        raise ValueError(f"Unknown projector type: {projector_type}")

    depth = int(match.group(1))
    total = linear_params(in_features, out_features)
    for _ in range(1, depth):
        total += linear_params(out_features, out_features)
    return total


def timestep_embedder_params(hidden_size: int, frequency_embedding_size: int = 256) -> int:
    return (
        linear_params(frequency_embedding_size, hidden_size)
        + linear_params(hidden_size, hidden_size)
    )


def rdt_block_params(hidden_size: int, num_heads: int) -> int:
    head_dim = hidden_size // num_heads

    self_attention = (
        linear_params(hidden_size, hidden_size * 3)
        + head_dim
        + head_dim
        + linear_params(hidden_size, hidden_size)
    )
    cross_attention = (
        linear_params(hidden_size, hidden_size)
        + linear_params(hidden_size, hidden_size * 2)
        + head_dim
        + head_dim
        + linear_params(hidden_size, hidden_size)
    )
    ffn = linear_params(hidden_size, hidden_size) + linear_params(hidden_size, hidden_size)
    rms_norms = hidden_size * 3
    return self_attention + cross_attention + ffn + rms_norms


def rdt_core_params(
    *,
    action_dim: int,
    pred_horizon: int,
    num_state_tokens: int,
    hidden_size: int,
    depth: int,
    num_heads: int,
    max_lang_cond_len: int,
    img_cond_len: int,
) -> dict[str, int]:
    x_pos_embed = (pred_horizon + num_state_tokens + 2) * hidden_size
    lang_cond_pos_embed = max_lang_cond_len * hidden_size
    img_cond_pos_embed = img_cond_len * hidden_size
    embedders = 2 * timestep_embedder_params(hidden_size)
    blocks = depth * rdt_block_params(hidden_size, num_heads)
    final_layer = (
        hidden_size
        + linear_params(hidden_size, hidden_size)
        + linear_params(hidden_size, action_dim)
    )
    return {
        "x_pos_embed": x_pos_embed,
        "lang_cond_pos_embed": lang_cond_pos_embed,
        "img_cond_pos_embed": img_cond_pos_embed,
        "timestep_and_freq_embedders": embedders,
        "rdt_blocks": blocks,
        "final_layer": final_layer,
    }


def stage2_formula_rows(
    config: dict[str, Any],
    stage1_config: dict[str, Any],
    *,
    vision_num_patches: int,
    use_stage1_condition: bool,
    stage1_total: Optional[int],
) -> list[CountRow]:
    common = config["common"]
    model_cfg = config["model"]
    rdt_cfg = model_cfg["rdt"]
    hidden_size = int(rdt_cfg["hidden_size"])
    img_cond_len = (
        int(common["img_history_size"])
        * int(common["num_cameras"])
        * int(vision_num_patches)
    )
    if use_stage1_condition:
        future_steps = int(
            stage1_config["model"].get(
                "future_steps",
                stage1_config.get("data", {}).get(
                    "future_steps",
                    stage1_config.get("data", {}).get("recipe", {}).get("future_steps", 1),
                ),
            )
        )
        num_state_tokens = 1 + max(1, future_steps)
    else:
        num_state_tokens = 1
    components = rdt_core_params(
        action_dim=int(common["state_dim"]),
        pred_horizon=int(common["action_chunk_size"]),
        num_state_tokens=num_state_tokens,
        hidden_size=hidden_size,
        depth=int(rdt_cfg["depth"]),
        num_heads=int(rdt_cfg["num_heads"]),
        max_lang_cond_len=int(config["dataset"]["tokenizer_max_length"]),
        img_cond_len=img_cond_len,
    )
    components["lang_adaptor"] = adapter_params(
        model_cfg["lang_adaptor"],
        int(model_cfg["lang_token_dim"]),
        hidden_size,
    )
    components["img_adaptor"] = adapter_params(
        model_cfg["img_adaptor"],
        int(model_cfg["img_token_dim"]),
        hidden_size,
    )
    components["state_adaptor"] = adapter_params(
        model_cfg["state_adaptor"],
        int(model_cfg["state_token_dim"]) * 2,
        hidden_size,
    )

    total_without_stage1 = sum(components.values())
    rows = [
        CountRow(
            name=(
                "stage2_formula_excluding_embedded_stage1"
                if use_stage1_condition
                else "stage2_formula"
            ),
            total=total_without_stage1,
            trainable=total_without_stage1,
            frozen=0,
            source="config formula",
            notes=f"img_cond_len={img_cond_len}, num_state_tokens={num_state_tokens}",
        )
    ]

    if use_stage1_condition:
        target_pose_adaptor = adapter_params(
            model_cfg.get("target_pose_adaptor", model_cfg["state_adaptor"]),
            int(stage1_config["model"].get("embed_dim", hidden_size)),
            hidden_size,
        )
        conditioned_total = total_without_stage1 + target_pose_adaptor
        rows.append(
            CountRow(
                name="stage2_conditioned_wrapper_excluding_embedded_stage1",
                total=conditioned_total,
                trainable=conditioned_total,
                frozen=0,
                source="config formula",
                notes=f"adds target_pose_adaptor={target_pose_adaptor:,}",
            )
        )
        if stage1_total is not None:
            rows.append(
                CountRow(
                    name="stage2_conditioned_wrapper_including_embedded_stage1",
                    total=conditioned_total + stage1_total,
                    trainable=conditioned_total,
                    frozen=stage1_total,
                    source="config formula + stage1 count",
                    notes="Stage 1 is frozen inside Stage1ConditionedRDTRunner.",
                )
            )

    return rows


def stage2_formula_count(
    config: dict[str, Any],
    stage1_config: dict[str, Any],
    *,
    vision_num_patches: int,
    use_stage1_condition: bool,
) -> tuple[int, str]:
    rows = stage2_formula_rows(
        config,
        stage1_config,
        vision_num_patches=vision_num_patches,
        use_stage1_condition=use_stage1_condition,
        stage1_total=None,
    )
    if use_stage1_condition:
        return rows[-1].total, rows[-1].notes
    return rows[0].total, rows[0].notes


def stage1_non_backbone_params(config: dict[str, Any]) -> int:
    model_cfg = config["model"]
    embed_dim = int(model_cfg["embed_dim"])
    pose_dim = int(model_cfg["pose_dim"])
    feedforward_dim = embed_dim * int(model_cfg.get("feedforward_mult", 8))
    num_layers = int(model_cfg["num_layers"])
    future_steps = max(
        1,
        int(
            model_cfg.get(
                "future_steps",
                config.get("data", {}).get(
                    "future_steps",
                    config.get("data", {}).get("recipe", {}).get("future_steps", 1),
                ),
            )
        ),
    )

    projections = (
        linear_params(int(model_cfg["image_feature_dim"]), embed_dim)
        + linear_params(int(model_cfg["language_feature_dim"]), embed_dim)
        + linear_params(pose_dim, embed_dim)
        + linear_params(embed_dim, embed_dim)
    )
    learned_tokens = (
        3 * embed_dim
        + (int(model_cfg.get("num_fusion_tokens", 32)) + future_steps) * embed_dim
    )
    transformer_layer = (
        linear_params(embed_dim, embed_dim * 3)
        + linear_params(embed_dim, embed_dim)
        + linear_params(embed_dim, feedforward_dim)
        + linear_params(feedforward_dim, embed_dim)
        + 4 * embed_dim
    )
    pose_head = (
        2 * embed_dim
        + linear_params(embed_dim, embed_dim)
        + linear_params(embed_dim, embed_dim)
        + linear_params(embed_dim, pose_dim)
    )
    frame_head = (
        embed_dim
        + linear_params(embed_dim, embed_dim)
        + linear_params(embed_dim, embed_dim)
        + linear_params(embed_dim, int(model_cfg["image_feature_dim"]))
    )
    return projections + learned_tokens + num_layers * transformer_layer + future_steps * (pose_head + frame_head)


def count_stage1(args: argparse.Namespace, config: dict[str, Any]) -> CountRow:
    inference_cfg = config.get("inference", {})
    checkpoint_name = (
        args.stage1_checkpoint_name
        or inference_cfg.get("checkpoint_name")
        or "last.ckpt"
    )
    checkpoint_path = resolve_checkpoint_path(
        args.stage1_checkpoint_path or inference_cfg.get("checkpoint_path"),
        args.stage1_checkpoint_dir or inference_cfg.get("checkpoint_dir"),
        checkpoint_name,
    )

    source = args.stage1_count_source
    if source == "auto":
        if checkpoint_path is not None:
            source = "checkpoint"
        elif args.allow_stage1_model_load:
            source = "load"
        else:
            source = "formula"

    if source == "checkpoint":
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No Stage 1 checkpoint was found. Provide --stage1-checkpoint-path "
                "or --stage1-checkpoint-dir, or use --stage1-count-source formula/load."
            )
        state, state_source = load_tensor_dict(checkpoint_path)
        total = count_state_tensors(state)
        return CountRow(
            name="stage1_checkpoint_tensors",
            total=total,
            source=f"{checkpoint_path} ({state_source})",
            notes="Counts checkpoint tensors; trainable/frozen split is unavailable.",
        )

    if source == "load":
        from stage1.config_utils import build_stage1_model_kwargs
        from stage1.model.stage1_transformer import Stage1Transformer

        model = Stage1Transformer(**build_stage1_model_kwargs(config))
        return module_count_row("stage1_loaded_model", model, "instantiated Stage1Transformer")

    total = stage1_non_backbone_params(config)
    freeze_backbones = bool(config["model"].get("freeze_backbones", True))
    trainable = total if freeze_backbones else None
    return CountRow(
        name="stage1_formula_non_backbone",
        total=total,
        trainable=trainable,
        frozen=None,
        source="config formula",
        notes=(
            "Excludes DINOv2, SigLIP, and T5 backbone parameters. Use a checkpoint "
            "or --stage1-count-source load for the full Stage 1 count."
        ),
    )


def stage2_checkpoint_rows(path: str) -> list[CountRow]:
    model_file = resolve_model_file(path)
    state, state_source = load_tensor_dict(model_file)
    total = count_state_tensors(state)
    rows = [
        CountRow(
            name="stage2_checkpoint_tensors",
            total=total,
            source=f"{model_file} ({state_source})",
            notes="Counts checkpoint tensors; trainable/frozen split is unavailable.",
        )
    ]
    embedded_stage1 = count_state_tensors(state, include_prefix="stage1_model.")
    if embedded_stage1:
        rows.append(
            CountRow(
                name="stage2_checkpoint_excluding_embedded_stage1",
                total=count_state_tensors(state, exclude_prefix="stage1_model."),
                source=f"{model_file} ({state_source})",
                notes=f"Detected embedded stage1_model tensors={embedded_stage1:,}.",
            )
        )
    return rows


def whole_model_row(
    *,
    stage1_row: Optional[CountRow],
    stage2_total: Optional[int],
    stage2_notes: str,
    use_stage1_condition: bool,
    stage2_checkpoint_row: Optional[CountRow],
    stage2_checkpoint_has_embedded_stage1: bool,
) -> CountRow:
    if stage2_checkpoint_row is not None:
        include_stage1 = stage1_row is not None and not stage2_checkpoint_has_embedded_stage1
        total = stage2_checkpoint_row.total + (stage1_row.total if include_stage1 else 0)
        return CountRow(
            name="whole_model",
            total=total,
            trainable=stage2_checkpoint_row.trainable,
            frozen=stage2_checkpoint_row.frozen,
            source=stage2_checkpoint_row.source,
            notes=(
                "Counted from Stage 2 checkpoint tensors."
                + (
                    " Embedded Stage 1 tensors were detected, so they are already included."
                    if stage2_checkpoint_has_embedded_stage1
                    else " Added Stage 1 count because no embedded Stage 1 tensors were detected."
                    if include_stage1
                    else ""
                )
            ),
        )

    if stage2_total is None:
        raise ValueError("Cannot compute whole model count without Stage 2.")

    if stage1_row is None:
        return CountRow(
            name="whole_model",
            total=stage2_total,
            trainable=stage2_total,
            frozen=0,
            source="config formula",
            notes=stage2_notes,
        )

    stage1_is_partial = stage1_row.name.endswith("_non_backbone")
    total = stage2_total + stage1_row.total
    if use_stage1_condition:
        trainable = stage2_total
        frozen = None if stage1_is_partial else stage1_row.total
    else:
        trainable = None if stage1_row.trainable is None else stage2_total + stage1_row.trainable
        frozen = stage1_row.frozen
    return CountRow(
        name="whole_model",
        total=total,
        trainable=trainable,
        frozen=frozen,
        source=f"config formula + {stage1_row.source}",
        notes=(
            (
                "Stage1ConditionedRDTRunner total = Stage 2 conditioned wrapper + Stage 1."
                if use_stage1_condition
                else "Whole total = Stage 2 + Stage 1."
            )
            + f" {stage2_notes}"
            + (
                " Stage 1 count excludes DINOv2/SigLIP/T5 backbones; pass a "
                "Stage 1 checkpoint or --stage1-count-source load for the exact "
                "whole-model total."
                if stage1_is_partial
                else (
                    " Stage 1 is frozen inside the wrapper."
                    if use_stage1_condition
                    else ""
                )
            )
        ),
    )


def print_table(rows: list[CountRow]) -> None:
    headers = ("name", "total", "trainable", "frozen", "source", "notes")
    table = []
    for row in rows:
        table.append(
            (
                row.name,
                format_count(row.total),
                format_count(row.trainable),
                format_count(row.frozen),
                row.source,
                row.notes,
            )
        )
    widths = [
        max(len(str(item[index])) for item in [headers, *table])
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def print_whole_count(row: CountRow) -> None:
    print(f"whole_model_params: {format_count(row.total)}")
    if row.notes:
        print(f"note: {row.notes}")


def main() -> None:
    args = parse_args()
    detail_rows: list[CountRow] = []

    stage1_config = load_yaml(args.stage1_config_path)
    stage2_config = load_yaml(args.stage2_config_path)
    stage1_row: Optional[CountRow] = None
    stage2_total: Optional[int] = None
    stage2_notes = ""
    stage2_checkpoint_row: Optional[CountRow] = None
    stage2_checkpoint_has_embedded_stage1 = False
    use_stage1_condition = False

    if not args.skip_stage1:
        stage1_row = count_stage1(args, stage1_config)
        detail_rows.append(stage1_row)

    if not args.skip_stage2:
        stage2_stage1_cfg = stage2_config.get("stage1", {})
        use_stage1_condition = args.use_stage1_condition or bool(
            stage2_stage1_cfg.get("enabled", False)
        )
        stage2_total, stage2_notes = stage2_formula_count(
            stage2_config,
            stage1_config,
            vision_num_patches=args.vision_num_patches,
            use_stage1_condition=use_stage1_condition,
        )
        detail_rows.extend(
            stage2_formula_rows(
                stage2_config,
                stage1_config,
                vision_num_patches=args.vision_num_patches,
                use_stage1_condition=use_stage1_condition,
                stage1_total=stage1_row.total if stage1_row is not None else None,
            )
        )
        if args.stage2_checkpoint_path:
            checkpoint_rows = stage2_checkpoint_rows(args.stage2_checkpoint_path)
            detail_rows.extend(checkpoint_rows)
            stage2_checkpoint_row = checkpoint_rows[0]
            stage2_checkpoint_has_embedded_stage1 = any(
                row.name == "stage2_checkpoint_excluding_embedded_stage1"
                for row in checkpoint_rows
            )

    rows = [
        whole_model_row(
            stage1_row=stage1_row,
            stage2_total=stage2_total,
            stage2_notes=stage2_notes,
            use_stage1_condition=use_stage1_condition,
            stage2_checkpoint_row=stage2_checkpoint_row,
            stage2_checkpoint_has_embedded_stage1=stage2_checkpoint_has_embedded_stage1,
        )
    ]
    if args.details:
        rows.extend(detail_rows)

    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    elif args.details:
        print_table(rows)
    else:
        print_whole_count(rows[0])


if __name__ == "__main__":
    main()
