#!/usr/bin/env python3
"""Validate a release configuration without downloading model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def validate(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    for section in ("common", "dataset", "model", "stage1"):
        if section not in config:
            errors.append(f"missing top-level section: {section}")
    if int(config.get("common", {}).get("action_chunk_size", 0)) < 1:
        errors.append("common.action_chunk_size must be positive")
    if bool(config.get("stage1", {}).get("enabled", False)):
        stage1_config = Path(config["stage1"].get("config_path", ""))
        if not stage1_config.is_absolute():
            repo_root = config_path.resolve().parent.parent
            stage1_config = repo_root / stage1_config
        if not stage1_config.is_file():
            errors.append(f"stage1 config does not exist: {stage1_config}")
        else:
            stage1 = yaml.safe_load(stage1_config.read_text(encoding="utf-8")) or {}
            model = stage1.get("model", {})
            data = stage1.get("data", {})
            future_steps = model.get(
                "future_steps",
                data.get("future_steps", data.get("recipe", {}).get("future_steps", 0)),
            )
            if int(future_steps) < 1:
                errors.append("Stage 1 future_steps must be positive")
            if int(model.get("pose_dim", 0)) != 16:
                errors.append("Stage 1 pose_dim must be 16 for flattened 4x4 poses")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Release configuration is valid: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

