import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from stage1.config_utils import (
    build_stage1_model_kwargs,
    load_stage1_config,
    resolve_stage1_checkpoint_path,
)
from stage1.datamodule import Stage1DataModule
from stage1.model.stage1_transformer import Stage1Transformer

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:  # pragma: no cover - optional dependency
    WandbLogger = None


def get_global_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 model.")
    parser.add_argument(
        "--config",
        default="stage1/config/stage1.yaml",
        help="Path to a Stage 1 YAML config.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint file or directory to resume from. Overrides training.resume_checkpoint_path.",
    )
    return parser.parse_args()


def build_logger(config: dict):
    logging_config = config.get("logging", {})
    save_dir = logging_config.get("save_dir", "logs")
    run_name = logging_config.get("run_name", "stage1")

    if logging_config.get("use_wandb", False) and WandbLogger is not None:
        if get_global_rank() != 0:
            return CSVLogger(save_dir=save_dir, name=f"{run_name}_rank{get_global_rank()}")

        wandb_mode = logging_config.get("mode", "offline")
        os.environ["WANDB_MODE"] = wandb_mode
        os.environ.setdefault("WANDB_START_METHOD", "thread")

        logger = WandbLogger(
            project=logging_config["project"],
            name=run_name,
            save_dir=save_dir,
            mode=wandb_mode,
            config=config,
        )
        return logger

    return CSVLogger(save_dir=save_dir, name=run_name)


def resolve_devices(devices: Any) -> Any:
    if isinstance(devices, str):
        stripped = devices.strip()
        if stripped.lower() == "auto":
            return stripped
        if "," in stripped:
            return [int(device_id.strip()) for device_id in stripped.split(",") if device_id.strip()]
        return int(stripped)
    return devices


def get_config_value(config: dict, key_path: str) -> Any:
    value: Any = config
    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing config key: {key_path}")
        value = value[key]
    return value


def sanitize_name_component(value: Any) -> str:
    text = str(value).strip().replace("/", "-")
    sanitized = "".join(character if character.isalnum() or character in {"-", "_", "."} else "-" for character in text)
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-") or "unknown"


def build_checkpoint_dir(config: dict) -> Path:
    training_config = config["training"]
    base_dir = Path(training_config.get("checkpoint_dir", "checkpoints/stage1"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir = datetime.now().strftime("%Y%m%d")

    name_components = []
    for key_path in training_config.get("checkpoint_name_params", []):
        key_name = key_path.split(".")[-1]
        key_value = get_config_value(config, key_path)
        name_components.append(f"{key_name}-{sanitize_name_component(key_value)}")

    run_name = "_".join(name_components + [timestamp])
    return base_dir / run_name / date_dir


def resolve_resume_checkpoint(config: dict, resume_from: str | None = None) -> Path | None:
    training_config = config["training"]
    if not training_config.get("resume", False) and resume_from is None:
        return None

    checkpoint_path = resume_from or training_config.get("resume_checkpoint_path")
    checkpoint_dir = training_config.get("resume_checkpoint_dir") or training_config.get("checkpoint_dir")
    checkpoint_name = training_config.get("resume_checkpoint_name", "last.ckpt")

    return resolve_stage1_checkpoint_path(
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
    )


def main() -> None:
    args = parse_args()
    config = load_stage1_config(args.config)
    data_config = config["data"]
    dataset_kwargs = dict(data_config.get("recipe", {}))
    dataset_kwargs.setdefault("future_steps", data_config.get("future_steps", 8))
    sharing_strategy = config["training"].get("sharing_strategy")
    if sharing_strategy:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    torch.set_float32_matmul_precision(config["training"].get("matmul_precision", "high"))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config["training"].get("cudnn_benchmark", True)

    dm = Stage1DataModule(
        data_path=data_config.get("path", "data/stage1_data/parsed_taco_data"),
        data_paths=data_config.get("paths"),
        dataset_kwargs=dataset_kwargs,
        data_sources=data_config.get("sources"),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        val_split=data_config.get("val_split", 0.1),
        seed=config["training"].get("seed", 42),
        prefetch_factor=config["training"].get("prefetch_factor", 4),
        persistent_workers=config["training"].get("persistent_workers", True),
        pin_memory=config["training"].get("pin_memory", False),
        prediction_target=data_config.get("prediction_target", "next_pose"),
        future_steps=int(data_config.get("future_steps", 8)),
    )

    model = Stage1Transformer(**build_stage1_model_kwargs(config))
    resume_checkpoint = resolve_resume_checkpoint(config, args.resume_from)
    if resume_checkpoint is not None:
        print(f"Resuming Stage 1 training from checkpoint: {resume_checkpoint}")

    checkpoint_dir = build_checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_config = config["training"]

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="stage1-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            every_n_epochs=training_config.get("checkpoint_every_n_epochs", 5),
            save_last=training_config.get("save_last_checkpoint", True),
            save_top_k=training_config.get("checkpoint_save_top_k", -1),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_devices = resolve_devices(training_config.get("devices", 1))
    is_multi_device = trainer_devices == "auto" or (
        isinstance(trainer_devices, int) and trainer_devices > 1
    ) or (isinstance(trainer_devices, (list, tuple)) and len(trainer_devices) > 1)

    trainer_strategy = training_config.get("strategy", "auto")
    if trainer_strategy == "auto" and is_multi_device and torch.cuda.is_available():
        trainer_strategy = "ddp"

    trainer = pl.Trainer(
        max_epochs=training_config["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_devices,
        num_nodes=training_config.get("num_nodes", 1),
        strategy=trainer_strategy,
        sync_batchnorm=training_config.get("sync_batchnorm", False),
        logger=build_logger(config),
        callbacks=callbacks,
        deterministic=training_config.get("deterministic", True),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.0),
        log_every_n_steps=training_config.get("log_every_n_steps", 10),
        precision=training_config.get("precision", "bf16-mixed"),
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch", 5),
    )

    trainer.fit(model, datamodule=dm, ckpt_path=str(resume_checkpoint) if resume_checkpoint else None)

    if WandbLogger is not None and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.finish()


if __name__ == "__main__":
    main()
