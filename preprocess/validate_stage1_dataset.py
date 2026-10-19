#!/usr/bin/env python3
"""Validate the artifact contract emitted by the Stage 1 pose pipeline.

The foundation-model stages (Grounded-SAM, SAM 3D, and FoundationPose) are
deliberately kept as external dependencies. This validator makes their output
contract explicit before a training job consumes it.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def validate_dataset(root: Path, future_steps: int = 8) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    sequence_count = 0
    sample_count = 0
    task_counts: dict[str, int] = {}

    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_sequence_count = 0
        for sequence_dir in sorted(task_dir.glob("seq_*")):
            pose_path = sequence_dir / "tool_poses.pkl"
            image_dir = sequence_dir / "rgb"
            mask_dir = sequence_dir / "masks"
            mesh_path = sequence_dir / "tool_mesh.obj"
            if not pose_path.is_file():
                errors.append(f"{sequence_dir}: missing tool_poses.pkl")
                continue
            if not image_dir.is_dir():
                errors.append(f"{sequence_dir}: missing rgb/")
                continue
            if not mask_dir.is_dir():
                warnings.append(f"{sequence_dir}: missing masks/ (required for preprocessing audit)")
            if not mesh_path.is_file():
                warnings.append(f"{sequence_dir}: missing tool_mesh.obj (SAM 3D output)")

            image_paths = sorted(
                path
                for path in image_dir.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            try:
                with pose_path.open("rb") as handle:
                    poses = np.asarray(pickle.load(handle), dtype=np.float32).squeeze()
            except Exception as exc:  # pragma: no cover - defensive reporting
                errors.append(f"{pose_path}: unable to read pose sequence: {exc}")
                continue

            if poses.ndim != 3 or poses.shape[1:] != (4, 4):
                errors.append(f"{pose_path}: expected (N, 4, 4), got {poses.shape}")
                continue
            if len(image_paths) < len(poses):
                errors.append(
                    f"{sequence_dir}: {len(image_paths)} RGB frames for {len(poses)} poses"
                )
            if len(poses) <= future_steps:
                errors.append(
                    f"{sequence_dir}: needs more than future_steps={future_steps} poses"
                )
                continue
            if not np.isfinite(poses).all():
                errors.append(f"{pose_path}: contains NaN or infinite pose values")

            instruction_path = next(
                (sequence_dir / name for name in ("instruction.txt", "task.txt", "language.txt")
                 if (sequence_dir / name).is_file()),
                None,
            )
            if instruction_path is None:
                warnings.append(f"{sequence_dir}: missing instruction text")
            usable = len(poses) - future_steps
            sample_count += usable
            task_sequence_count += 1
            sequence_count += 1

        if task_sequence_count:
            task_counts[task_dir.name] = task_sequence_count

    return {
        "root": str(root),
        "future_steps": future_steps,
        "tasks": task_counts,
        "sequence_count": sequence_count,
        "usable_sample_count": sample_count,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Stage 1 dataset root")
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {args.root}")
    report = validate_dataset(args.root, max(1, args.future_steps))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Stage 1 dataset: {report['sequence_count']} sequences, "
              f"{report['usable_sample_count']} usable samples")
        for message in report["warnings"]:
            print(f"WARNING: {message}")
        for message in report["errors"]:
            print(f"ERROR: {message}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

