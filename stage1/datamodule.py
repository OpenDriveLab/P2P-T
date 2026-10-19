from __future__ import annotations

from typing import Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from stage1.dataset import Stage1Dataset, Stage1DeltaPoseDataset, Stage1TargetPoseDataset


class Stage1DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str | Sequence[str] = "data/stage1_data/parsed_taco_data",
        data_paths: Optional[Sequence[str]] = None,
        dataset_kwargs: Optional[dict] = None,
        data_sources: Optional[Sequence[dict]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = False,
        prediction_target: str = "next_pose",
        future_steps: int = 8,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_paths = data_paths
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.data_sources = list(data_sources or [])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and num_workers > 0
        self.pin_memory = pin_memory
        self.prediction_target = prediction_target
        self.future_steps = max(1, int(future_steps))
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def _dataloader_kwargs(self, shuffle: bool) -> dict:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _build_dataset(self) -> torch.utils.data.Dataset:
        dataset_map = {
            "next_pose": Stage1Dataset,
            "target_pose": Stage1TargetPoseDataset,
            "delta_pose": Stage1DeltaPoseDataset,
        }
        try:
            dataset_cls = dataset_map[self.prediction_target]
        except KeyError as exc:
            valid_targets = ", ".join(sorted(dataset_map))
            raise ValueError(
                f"Unsupported prediction_target '{self.prediction_target}'. "
                f"Expected one of: {valid_targets}."
            ) from exc
        if self.data_sources:
            datasets = []
            for source in self.data_sources:
                source_kwargs = dict(source.get("recipe", {}))
                source_kwargs.setdefault("future_steps", self.future_steps)
                datasets.append(
                    dataset_cls(
                        root_dir=source.get("path", self.data_path),
                        root_dirs=source.get("paths"),
                        **source_kwargs,
                    )
                )
            return ConcatDataset(datasets)

        dataset_kwargs = dict(self.dataset_kwargs)
        dataset_kwargs.setdefault("future_steps", self.future_steps)
        return dataset_cls(
            root_dir=self.data_path,
            root_dirs=self.data_paths,
            **dataset_kwargs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        full_dataset = self._build_dataset()
        if len(full_dataset) == 0:
            roots = self.data_sources or self.data_paths or self.data_path
            raise RuntimeError(f"No valid samples were found under {roots}.")

        if len(full_dataset) == 1:
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset
            self.test_dataset = full_dataset
            return

        val_size = max(1, int(round(len(full_dataset) * self.val_split)))
        val_size = min(val_size, len(full_dataset) - 1)
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self._dataloader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self._dataloader_kwargs(shuffle=False))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self._dataloader_kwargs(shuffle=False))
