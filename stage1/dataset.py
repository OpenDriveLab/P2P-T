from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def flatten_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose matrix with shape (4, 4), got {pose.shape}")
    return pose.reshape(-1).copy()


def compute_delta_pose(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    if current_pose.shape != (4, 4) or target_pose.shape != (4, 4):
        raise ValueError(
            "Expected current_pose and target_pose to both have shape (4, 4), "
            f"got {current_pose.shape} and {target_pose.shape}"
        )
    delta_pose = np.linalg.inv(current_pose) @ target_pose
    return delta_pose.astype(np.float32, copy=False)


def apply_delta_pose(current_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    delta_pose = np.asarray(delta_pose, dtype=np.float32)
    if current_pose.shape != (4, 4) or delta_pose.shape != (4, 4):
        raise ValueError(
            "Expected current_pose and delta_pose to both have shape (4, 4), "
            f"got {current_pose.shape} and {delta_pose.shape}"
        )
    next_pose = current_pose @ delta_pose
    return next_pose.astype(np.float32, copy=False)


def _format_noun_phrase(text: str, add_article: bool = False) -> str:
    phrase = text.replace("_", " ").strip()
    if not phrase:
        return phrase

    lowered = phrase.lower()
    if lowered.startswith(("a ", "an ", "the ")):
        return phrase
    if add_article:
        return f"the {phrase}"
    return phrase


def format_task_name_as_instruction(task_name: str) -> str:
    normalized = task_name.strip().replace("-", "_")
    if not normalized:
        return ""

    parts = [part for part in normalized.split("_") if part]
    if len(parts) >= 3:
        action = parts[0].replace("_", " ")
        tool = _format_noun_phrase(parts[1])
        target = _format_noun_phrase(" ".join(parts[2:]), add_article=True)
        return f"use {tool} to {action} {target}"

    phrase = normalized.replace("_", " ")
    return phrase


class BaseStage1Dataset(Dataset):
    """Base dataset for stage 1 pose prediction tasks."""

    def __init__(
        self,
        root_dir: str | Sequence[str] = "data/stage1_data/parsed_taco_data",
        root_dirs: Optional[Sequence[str]] = None,
        transform: Optional[transforms.Compose] = None,
        task_names: Optional[Sequence[str]] = None,
        sequence_globs: Optional[Sequence[str]] = None,
        pose_filenames: Optional[Sequence[str]] = None,
        image_dirs: Optional[Sequence[str]] = None,
        image_globs: Optional[Sequence[str]] = None,
        image_filename_format: str = "{frame_idx:06d}.png",
        instruction_template: Optional[str] = None,
        task_instructions: Optional[Dict[str, str]] = None,
        frame_stride: int = 1,
        future_steps: int = 1,
        max_sequences: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        self.root_dirs = self._normalize_root_dirs(root_dir, root_dirs)
        self.task_names = set(task_names) if task_names else None
        self.sequence_globs = list(sequence_globs or ["seq_*"])
        self.pose_filenames = list(pose_filenames or ["tool_poses.pkl"])
        self.image_dirs = list(image_dirs or ["rgb"])
        self.image_globs = list(image_globs or ["*.png", "*.jpg", "*.jpeg"])
        self.image_filename_format = image_filename_format
        self.instruction_template = instruction_template
        self.task_instructions = dict(task_instructions or {})
        self.frame_stride = max(1, int(frame_stride))
        self.future_steps = max(1, int(future_steps))
        self.max_sequences = max_sequences
        self.max_samples_per_sequence = max_samples_per_sequence
        self.strict = strict
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.samples: List[Dict[str, object]] = []

        missing_roots = [str(root_dir) for root_dir in self.root_dirs if not root_dir.is_dir()]
        if missing_roots and self.strict:
            raise FileNotFoundError(
                "Stage1 dataset directory was not found: "
                + ", ".join(missing_roots)
            )
        self.root_dirs = [root_dir for root_dir in self.root_dirs if root_dir.is_dir()]
        if not self.root_dirs:
            raise FileNotFoundError(
                "No Stage1 dataset directories were found. Checked: "
                + ", ".join(missing_roots)
            )

        self._build_index()

    @staticmethod
    def _normalize_root_dirs(
        root_dir: str | Sequence[str],
        root_dirs: Optional[Sequence[str]],
    ) -> List[Path]:
        raw_roots: Sequence[str]
        if root_dirs is not None:
            raw_roots = root_dirs
        elif isinstance(root_dir, (list, tuple)):
            raw_roots = root_dir
        else:
            raw_roots = [root_dir]
        return [Path(path).expanduser() for path in raw_roots]

    @staticmethod
    def _load_pose_sequence(tool_poses_path: str) -> np.ndarray:
        with open(tool_poses_path, "rb") as handle:
            tool_poses = np.asarray(pickle.load(handle), dtype=np.float32)

        # Stored poses come in as (1, L, 4, 4). Remove the singleton batch axis
        # so we can index the temporal dimension directly.
        tool_poses = np.squeeze(tool_poses)

        if tool_poses.ndim != 3 or tool_poses.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected tool poses with shape (L, 4, 4) after squeeze, got {tool_poses.shape}"
            )

        return tool_poses

    @staticmethod
    def _sort_image_paths(paths: Sequence[Path]) -> List[Path]:
        def sort_key(path: Path) -> tuple[int, int | str, str]:
            try:
                return (0, int(path.stem), path.name)
            except ValueError:
                return (1, path.stem, path.name)

        return sorted(paths, key=sort_key)

    def _find_first_file(self, seq_path: Path, filenames: Sequence[str]) -> Optional[Path]:
        for filename in filenames:
            path = seq_path / filename
            if path.is_file():
                return path
        return None

    def _find_first_dir(self, seq_path: Path, dirnames: Sequence[str]) -> Optional[Path]:
        for dirname in dirnames:
            path = seq_path / dirname
            if path.is_dir():
                return path
        return None

    def _list_image_paths(self, image_dir: Path) -> List[Path]:
        image_paths: List[Path] = []
        for pattern in self.image_globs:
            image_paths.extend(image_dir.glob(pattern))
        return self._sort_image_paths({path for path in image_paths if path.is_file()})

    def _resolve_image_path(
        self,
        image_dir: Path,
        image_paths: Sequence[Path],
        frame_idx: int,
    ) -> Optional[str]:
        try:
            formatted_name = self.image_filename_format.format(
                frame_idx=frame_idx,
                index=frame_idx,
            )
        except (KeyError, ValueError):
            formatted_name = f"{frame_idx:06d}.png"

        formatted_path = image_dir / formatted_name
        if formatted_path.is_file():
            return str(formatted_path)

        if frame_idx < len(image_paths):
            return str(image_paths[frame_idx])

        return None

    def _iter_sequence_paths(self, task_path: Path) -> List[Path]:
        seq_paths: List[Path] = []
        for pattern in self.sequence_globs:
            seq_paths.extend(path for path in task_path.glob(pattern) if path.is_dir())
        if not seq_paths and "*" not in "".join(self.sequence_globs):
            seq_paths = [path for path in task_path.iterdir() if path.is_dir()]
        return sorted(set(seq_paths))

    def _build_index(self) -> None:
        num_sequences = 0
        for root_dir in self.root_dirs:
            for task_path in sorted(path for path in root_dir.iterdir() if path.is_dir()):
                task_name = task_path.name
                if self.task_names is not None and task_name not in self.task_names:
                    continue

                for seq_path in self._iter_sequence_paths(task_path):
                    if self.max_sequences is not None and num_sequences >= self.max_sequences:
                        return

                    tool_poses_path = self._find_first_file(seq_path, self.pose_filenames)
                    image_dir = self._find_first_dir(seq_path, self.image_dirs)
                    if tool_poses_path is None or image_dir is None:
                        continue

                    tool_poses = self._load_pose_sequence(str(tool_poses_path))

                    if tool_poses.shape[0] < 2:
                        continue

                    image_paths = self._list_image_paths(image_dir)
                    if not image_paths:
                        continue

                    self._build_sequence_samples(
                        str(seq_path),
                        image_dir,
                        image_paths,
                        tool_poses,
                    )
                    num_sequences += 1

    def _build_sequence_samples(
        self,
        seq_path: str,
        image_dir: Path,
        image_paths: Sequence[Path],
        tool_poses: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def _load_instruction(self, seq_path: str) -> str:
        candidates = [
            "instruction.txt",
            "instruction.md",
            "task.txt",
            "language.txt",
            "text.txt",
        ]
        for filename in candidates:
            path = os.path.join(seq_path, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read().strip()
                if text:
                    return text
        task_name = os.path.basename(os.path.dirname(seq_path))
        if task_name in self.task_instructions:
            return self.task_instructions[task_name]
        if self.instruction_template:
            formatted_task = format_task_name_as_instruction(task_name)
            return self.instruction_template.format(
                task=formatted_task,
                task_name=task_name,
            )
        return format_task_name_as_instruction(task_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.transform(image)

        future_images = []
        for image_path in sample.get("future_image_paths", [sample["image_path"]]):
            future_image = Image.open(image_path).convert("RGB")
            future_images.append(self.transform(future_image))
        future_image_tensor = torch.stack(future_images, dim=0)
        future_pose_tensor = torch.from_numpy(
            np.stack(sample.get("future_poses", [sample["target_pose"]]), axis=0)
        ).float()

        return {
            "image": image_tensor,
            "current_tool_pose": torch.from_numpy(sample["current_tool_pose"]).float(),
            "instruction": sample["instruction"],
            "target_pose": torch.from_numpy(sample["target_pose"]).float(),
            "future_images": future_image_tensor,
            "future_poses": future_pose_tensor,
        }


class Stage1Dataset(BaseStage1Dataset):
    """Dataset for Stage 1 future pose and visual-target prediction."""

    def _build_sequence_samples(
        self,
        seq_path: str,
        image_dir: Path,
        image_paths: Sequence[Path],
        tool_poses: np.ndarray,
    ) -> None:
        instruction = self._load_instruction(seq_path)
        num_source_frames = tool_poses.shape[0] - self.future_steps
        sample_count = 0

        for frame_idx in range(0, num_source_frames, self.frame_stride):
            if (
                self.max_samples_per_sequence is not None
                and sample_count >= self.max_samples_per_sequence
            ):
                break
            image_path = self._resolve_image_path(image_dir, image_paths, frame_idx)
            if image_path is None:
                continue
            future_image_paths = [
                self._resolve_image_path(image_dir, image_paths, frame_idx + offset)
                for offset in range(1, self.future_steps + 1)
            ]
            if any(path is None for path in future_image_paths):
                continue

            self.samples.append(
                {
                    "image_path": image_path,
                    "current_tool_pose": flatten_pose_matrix(tool_poses[frame_idx]),
                    "target_pose": flatten_pose_matrix(tool_poses[frame_idx + 1]),
                    "future_poses": [
                        flatten_pose_matrix(tool_poses[frame_idx + offset])
                        for offset in range(1, self.future_steps + 1)
                    ],
                    "future_image_paths": future_image_paths,
                    "instruction": instruction,
                    "sequence_path": seq_path,
                    "frame_idx": frame_idx,
                }
            )
            sample_count += 1


class Stage1TargetPoseDataset(BaseStage1Dataset):
    """Dataset for stage 1 final-target pose prediction."""

    def _build_sequence_samples(
        self,
        seq_path: str,
        image_dir: Path,
        image_paths: Sequence[Path],
        tool_poses: np.ndarray,
    ) -> None:
        instruction = self._load_instruction(seq_path)
        target_pose = flatten_pose_matrix(tool_poses[-1])
        num_source_frames = tool_poses.shape[0] - 1
        sample_count = 0

        for frame_idx in range(0, num_source_frames, self.frame_stride):
            if (
                self.max_samples_per_sequence is not None
                and sample_count >= self.max_samples_per_sequence
            ):
                break
            image_path = self._resolve_image_path(image_dir, image_paths, frame_idx)
            if image_path is None:
                continue
            future_image_paths = [
                self._resolve_image_path(image_dir, image_paths, frame_idx + offset)
                for offset in range(1, self.future_steps + 1)
            ]
            if any(path is None for path in future_image_paths):
                continue

            self.samples.append(
                {
                    "image_path": image_path,
                    "current_tool_pose": flatten_pose_matrix(tool_poses[frame_idx]),
                    "target_pose": target_pose,
                    "future_poses": [target_pose for _ in range(self.future_steps)],
                    "future_image_paths": future_image_paths,
                    "instruction": instruction,
                    "sequence_path": seq_path,
                    "frame_idx": frame_idx,
                }
            )
            sample_count += 1


class Stage1DeltaPoseDataset(BaseStage1Dataset):
    """Dataset for stage 1 relative next-pose prediction."""

    def _build_sequence_samples(
        self,
        seq_path: str,
        image_dir: Path,
        image_paths: Sequence[Path],
        tool_poses: np.ndarray,
    ) -> None:
        instruction = self._load_instruction(seq_path)
        num_source_frames = tool_poses.shape[0] - self.future_steps
        sample_count = 0

        for frame_idx in range(0, num_source_frames, self.frame_stride):
            if (
                self.max_samples_per_sequence is not None
                and sample_count >= self.max_samples_per_sequence
            ):
                break
            image_path = self._resolve_image_path(image_dir, image_paths, frame_idx)
            if image_path is None:
                continue

            current_pose = tool_poses[frame_idx]
            next_pose = tool_poses[frame_idx + 1]
            delta_pose = compute_delta_pose(current_pose, next_pose)
            future_image_paths = [
                self._resolve_image_path(image_dir, image_paths, frame_idx + offset)
                for offset in range(1, self.future_steps + 1)
            ]
            if any(path is None for path in future_image_paths):
                continue
            self.samples.append(
                {
                    "image_path": image_path,
                    "current_tool_pose": flatten_pose_matrix(current_pose),
                    "target_pose": flatten_pose_matrix(delta_pose),
                    "future_poses": [
                        flatten_pose_matrix(
                            compute_delta_pose(current_pose, tool_poses[frame_idx + offset])
                        )
                        for offset in range(1, self.future_steps + 1)
                    ],
                    "future_image_paths": future_image_paths,
                    "instruction": instruction,
                    "sequence_path": seq_path,
                    "frame_idx": frame_idx,
                }
            )
            sample_count += 1


if __name__ == "__main__":
    dataset = Stage1Dataset()
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Image shape:", sample["image"].shape)
        print("Instruction:", sample["instruction"] or "<empty>")
        print("Current tool pose shape:", sample["current_tool_pose"].shape)
        print("Target pose shape:", sample["target_pose"].shape)
