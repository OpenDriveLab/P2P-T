from __future__ import annotations

from typing import Iterable, Optional, Sequence

import os
import torch
import torch.nn as nn
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as load_safetensors_file

from models.rdt_runner import RDTRunner
from stage1.config_utils import (
    build_stage1_model_kwargs,
    load_stage1_config,
    resolve_stage1_checkpoint_path,
    resolve_torch_dtype,
    supports_autocast,
)
from stage1.model.stage1_transformer import Stage1Transformer


class Stage1ConditionedRDTRunner(RDTRunner):
    def __init__(
        self,
        *,
        action_dim,
        pred_horizon,
        config,
        lang_token_dim,
        img_token_dim,
        state_token_dim,
        max_lang_cond_len,
        img_cond_len,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16,
        stage1_config_path: str = "stage1/config/stage1.yaml",
        stage1_checkpoint_path: Optional[str] = None,
        stage1_checkpoint_dir: Optional[str] = None,
        stage1_checkpoint_name: str = "last.ckpt",
        stage1_target_camera_index: int = 0,
    ):
        stage1_config = load_stage1_config(stage1_config_path)
        stage1_future_steps = int(
            stage1_config["model"].get(
                "future_steps",
                stage1_config.get("data", {}).get(
                    "future_steps",
                    stage1_config.get("data", {}).get("recipe", {}).get("future_steps", 1),
                ),
            )
        )
        if stage1_future_steps < 1:
            raise ValueError("Stage 1 future_steps must be at least 1.")
        super().__init__(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            config=config,
            lang_token_dim=lang_token_dim,
            img_token_dim=img_token_dim,
            state_token_dim=state_token_dim,
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
            num_state_tokens=1 + stage1_future_steps,
        )
        self.stage1_config_path = stage1_config_path
        self.stage1_checkpoint_path = stage1_checkpoint_path
        self.stage1_checkpoint_dir = stage1_checkpoint_dir
        self.stage1_checkpoint_name = stage1_checkpoint_name
        self.stage1_target_camera_index = stage1_target_camera_index

        self.stage1_config = stage1_config
        self.stage1_pose_dim = int(self.stage1_config["model"]["pose_dim"])
        self.stage1_embed_dim = int(self.stage1_config["model"].get("embed_dim", self.stage1_pose_dim))
        self.stage1_future_steps = stage1_future_steps
        self.num_stage1_pose_tokens = stage1_future_steps
        self._stage1_runtime_device = torch.device("cpu")
        self._stage1_runtime_dtype = torch.float32
        adaptor_type = config.get("target_pose_adaptor", config["state_adaptor"])
        self.target_pose_adaptor = self.build_condition_adapter(
            adaptor_type,
            in_features=self.stage1_embed_dim,
            out_features=config["rdt"]["hidden_size"],
        )
        self.pose_fallback_adaptor = nn.Linear(
            self.stage1_pose_dim * 2, self.stage1_embed_dim
        )
        self.stage1_model = self._build_stage1_model()
        self._freeze_stage1_model()

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[str],
        force_download: bool,
        proxies: Optional[dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            safetensors_path = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            pickle_path = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
            checkpoint_path = safetensors_path if os.path.isfile(safetensors_path) else pickle_path
        else:
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except EntryNotFoundError:
                checkpoint_path = hf_hub_download(
                    repo_id=model_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )

        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_safetensors_file(checkpoint_path, device=map_location)
        else:
            state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]

        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        incompatible_keys = sorted(set(state_dict) - set(compatible_state))
        model.load_state_dict(compatible_state, strict=False)
        if incompatible_keys:
            print(
                "Loaded compatible pretrained RDT weights; initialized "
                f"{len(incompatible_keys)} unmatched Stage1-conditioned parameters from scratch."
            )
        return model

    def _build_stage1_model(self) -> Stage1Transformer:
        checkpoint_path = None
        try:
            checkpoint_path = resolve_stage1_checkpoint_path(
                checkpoint_path=self.stage1_checkpoint_path or self.stage1_config.get("inference", {}).get("checkpoint_path"),
                checkpoint_dir=self.stage1_checkpoint_dir or self.stage1_config.get("inference", {}).get("checkpoint_dir"),
                checkpoint_name=self.stage1_checkpoint_name,
            )
        except FileNotFoundError:
            checkpoint_path = None

        if checkpoint_path is not None:
            model = Stage1Transformer.load_from_checkpoint(
                str(checkpoint_path),
                map_location="cpu",
                strict=False,
                **build_stage1_model_kwargs(self.stage1_config),
            )
        else:
            model = Stage1Transformer(**build_stage1_model_kwargs(self.stage1_config))
        model.eval()
        return model

    def _freeze_stage1_model(self) -> None:
        self.stage1_model.requires_grad_(False)
        self.stage1_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1_model.eval()
        return self

    def _move_stage1_model_to(self, device: torch.device) -> None:
        device = torch.device(device)
        target_dtype = resolve_torch_dtype(self.stage1_config.get("inference", {}).get("model_dtype"))
        if target_dtype is not None and device.type == "cuda":
            dtype = target_dtype
        else:
            dtype = self._stage1_runtime_dtype
        if self._stage1_runtime_device == device and self._stage1_runtime_dtype == dtype:
            return
        self.stage1_model.to(device=device, dtype=dtype)
        self._stage1_runtime_device = device
        self._stage1_runtime_dtype = dtype

    def _default_current_tool_poses(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.stage1_pose_dim == 16:
            pose = torch.eye(4, device=device, dtype=dtype).reshape(1, 16)
            return pose.expand(batch_size, -1).contiguous()
        return torch.zeros(batch_size, self.stage1_pose_dim, device=device, dtype=dtype)

    def _predict_pose_latents(
        self,
        stage1_images: torch.Tensor,
        current_tool_poses: Optional[torch.Tensor],
        instructions: Sequence[str],
    ) -> torch.Tensor:
        if stage1_images is None or instructions is None:
            raise ValueError(
                "Stage1ConditionedRDTRunner requires stage1_images and instructions."
            )

        device = stage1_images.device
        if current_tool_poses is None:
            current_tool_poses = self._default_current_tool_poses(
                batch_size=stage1_images.shape[0],
                device=device,
                dtype=torch.float32,
            )
        else:
            current_tool_poses = current_tool_poses.to(device=device, dtype=torch.float32)

        self._move_stage1_model_to(device)
        autocast_dtype = resolve_torch_dtype(self.stage1_config.get("inference", {}).get("autocast_dtype"))
        autocast_enabled = supports_autocast(device, autocast_dtype)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = self.stage1_model(
                    stage1_images,
                    current_tool_poses,
                    list(instructions),
                    return_dict=True,
                )
        pose_latents = outputs.get("pose_latents") if isinstance(outputs, dict) else None
        if pose_latents is None:
            # Compatibility path for one-step checkpoints released before the
            # future-token interface was introduced.
            predicted_pose = outputs.get("predicted_poses", outputs) if isinstance(outputs, dict) else outputs
            if predicted_pose.ndim == 2:
                predicted_pose = predicted_pose.unsqueeze(1)
            pose_latents = self.pose_fallback_adaptor(
                torch.cat([predicted_pose, torch.ones_like(predicted_pose)], dim=-1)
            )
        return pose_latents.to(device=device, dtype=current_tool_poses.dtype)

    def _encode_target_pose_condition(
        self,
        target_poses: Optional[torch.Tensor] = None,
        stage1_images: Optional[torch.Tensor] = None,
        current_tool_poses: Optional[torch.Tensor] = None,
        instructions: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if target_poses is None:
            target_pose_tokens = self._predict_pose_latents(
                stage1_images, current_tool_poses, instructions
            )
        else:
            if target_poses.ndim == 2:
                target_poses = target_poses.unsqueeze(1)
            if target_poses.shape[-1] == self.stage1_pose_dim:
                target_pose_tokens = self.pose_fallback_adaptor(
                    torch.cat([target_poses, torch.ones_like(target_poses)], dim=-1)
                )
            elif target_poses.shape[-1] == self.stage1_embed_dim:
                target_pose_tokens = target_poses
            else:
                raise ValueError(
                    "target_poses must contain Stage 1 pose vectors or pose latents; "
                    f"got final dimension {target_poses.shape[-1]}."
                )
        adaptor_dtype = next(self.target_pose_adaptor.parameters()).dtype
        target_pose_tokens = target_pose_tokens.to(dtype=adaptor_dtype)
        return self.target_pose_adaptor(target_pose_tokens)

    def conditional_sample(
        self,
        lang_cond,
        lang_attn_mask,
        img_cond,
        state_traj,
        action_mask,
        ctrl_freqs,
        pose_condition=None,
    ):
        if pose_condition is None:
            raise ValueError("pose_condition must be provided for Stage1-conditioned sampling.")
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([pose_condition, state_traj, action_traj], dim=1)
            model_output = self.model(
                state_action_traj,
                ctrl_freqs,
                t.unsqueeze(-1).to(device),
                lang_cond,
                img_cond,
                lang_mask=lang_attn_mask,
            )
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        return noisy_action * action_mask

    def compute_loss(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_gt,
        action_mask,
        ctrl_freqs,
        target_poses=None,
        stage1_images=None,
        current_tool_poses=None,
        instructions=None,
    ) -> torch.Tensor:
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        masked_state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, masked_state_tokens)
        pose_condition = self._encode_target_pose_condition(
            target_poses=target_poses,
            stage1_images=stage1_images,
            current_tool_poses=current_tool_poses,
            instructions=instructions,
        ).to(dtype=state_traj.dtype)
        action_traj = torch.cat([noisy_action, action_mask.expand(-1, noisy_action.shape[1], -1)], dim=2)
        action_traj = self.state_adaptor(action_traj)
        state_action_traj = torch.cat([pose_condition, state_traj, action_traj], dim=1)
        pred = self.model(
            state_action_traj,
            ctrl_freqs,
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_attn_mask,
        )
        target = noise if self.prediction_type == "epsilon" else action_gt
        if self.prediction_type not in {"epsilon", "sample"}:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")
        return torch.nn.functional.mse_loss(pred, target)

    def predict_action(
        self,
        lang_tokens,
        lang_attn_mask,
        img_tokens,
        state_tokens,
        action_mask,
        ctrl_freqs,
        target_poses=None,
        stage1_images=None,
        current_tool_poses=None,
        instructions=None,
    ):
        masked_state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, masked_state_tokens)
        pose_condition = self._encode_target_pose_condition(
            target_poses=target_poses,
            stage1_images=stage1_images,
            current_tool_poses=current_tool_poses,
            instructions=instructions,
        ).to(dtype=state_traj.dtype)
        return self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            state_traj,
            action_mask,
            ctrl_freqs,
            pose_condition=pose_condition,
        )
