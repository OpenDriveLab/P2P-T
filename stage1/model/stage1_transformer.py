import time
import math
from pathlib import Path
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from diffusers.optimization import get_scheduler
from transformers import AutoProcessor, AutoTokenizer, SiglipVisionModel, T5EncoderModel

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_dinov2_from_hub(dino_repo: str, dino_model_name: str) -> nn.Module:
    repo_path = Path(dino_repo).expanduser()
    if repo_path.exists():
        return torch.hub.load(str(repo_path), dino_model_name, pretrained=True, source="local")

    try:
        return torch.hub.load(
            dino_repo,
            dino_model_name,
            pretrained=True,
            trust_repo=True,
            skip_validation=True,
        )
    except TypeError:
        return torch.hub.load(dino_repo, dino_model_name, pretrained=True)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        dino_repo: str = "facebookresearch/dinov2",
        dino_model_name: str = "dinov2_vitb14_reg",
        siglip_model_name: str = "google/siglip-base-patch16-224",
        dino_feature_dim: int = 768,
        siglip_feature_dim: int = 768,
        freeze_backbones: bool = True,
        backbone_image_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.dino_feature_dim = dino_feature_dim
        self.siglip_feature_dim = siglip_feature_dim
        self.freeze_backbones = freeze_backbones
        self.backbone_image_size = backbone_image_size

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        self.dinov2 = load_dinov2_from_hub(dino_repo, dino_model_name)
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
        self.siglip = SiglipVisionModel.from_pretrained(siglip_model_name)
        image_processor = self.siglip_processor.image_processor
        siglip_mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        siglip_std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        self.register_buffer(
            "siglip_mean",
            torch.tensor(siglip_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_std",
            torch.tensor(siglip_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

        if freeze_backbones:
            self.dinov2.requires_grad_(False)
            self.siglip.requires_grad_(False)
            self.dinov2.eval()
            self.siglip.eval()

    def _encode_dino(self, image: torch.Tensor) -> torch.Tensor:
        dino_input = self.dino_transform(image)
        dino_features = self.dinov2.forward_features(dino_input)
        return dino_features["x_norm_patchtokens"].mean(dim=1)

    def _encode_siglip(self, image: torch.Tensor) -> torch.Tensor:
        # The image is already resized for the vision backbones, so the
        # remaining processor work is just channel normalization. Doing that
        # directly on-device avoids a CPU round-trip for every batch.
        pixel_values = (image - self.siglip_mean) / self.siglip_std
        siglip_outputs = self.siglip(pixel_values=pixel_values)
        return siglip_outputs.pooler_output

    def _prepare_backbone_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == self.backbone_image_size:
            return image
        target_h, target_w = self.backbone_image_size
        _, channels, height, width = image.shape
        scale = min(target_h / height, target_w / width)
        resized_h = max(1, min(target_h, math.floor(height * scale)))
        resized_w = max(1, min(target_w, math.floor(width * scale)))

        resized = F.interpolate(
            image,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # Letterbox-pad to the backbone input size so we preserve the full field
        # of view instead of distorting wide observations into a square.
        padded = image.new_zeros((image.shape[0], channels, target_h, target_w))
        pad_top = (target_h - resized_h) // 2
        pad_left = (target_w - resized_w) // 2
        padded[:, :, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
        return padded

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._prepare_backbone_image(image)
        if self.freeze_backbones:
            with torch.no_grad():
                dino_features = self._encode_dino(image)
                siglip_features = self._encode_siglip(image)
        else:
            dino_features = self._encode_dino(image)
            siglip_features = self._encode_siglip(image)
        return torch.cat([dino_features, siglip_features], dim=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbones:
            self.dinov2.eval()
            self.siglip.eval()
        return self


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        text_model_name: str = "t5-base",
        freeze_backbone: bool = True,
        cache_embeddings: bool = True,
        max_text_length: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.model = T5EncoderModel.from_pretrained(text_model_name)
        self.freeze_backbone = freeze_backbone
        self.cache_embeddings = cache_embeddings and freeze_backbone
        self.max_text_length = max_text_length
        self._embedding_cache: dict[str, torch.Tensor] = {}

        if freeze_backbone:
            self.model.requires_grad_(False)
            self.model.eval()

    def _encode_text_batch(self, texts: List[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.model(**encoded)
        else:
            outputs = self.model(**encoded)

        hidden_states = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        return masked_hidden.sum(dim=1) / token_counts

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        normalized_texts = [text if text else "" for text in texts]
        if not self.cache_embeddings:
            return self._encode_text_batch(normalized_texts, device)

        missing_texts: List[str] = []
        for text in normalized_texts:
            if text not in self._embedding_cache:
                missing_texts.append(text)

        if missing_texts:
            unique_missing_texts = list(dict.fromkeys(missing_texts))
            missing_embeddings = self._encode_text_batch(unique_missing_texts, device)
            for text, embedding in zip(unique_missing_texts, missing_embeddings):
                self._embedding_cache[text] = embedding.detach().cpu()

        return torch.stack(
            [self._embedding_cache[text].to(device=device) for text in normalized_texts],
            dim=0,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self


class Stage1Transformer(pl.LightningModule):
    def __init__(
        self,
        image_feature_dim: int = 1536,
        language_feature_dim: int = 768,
        pose_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 8,
        embed_dim: int = 2048,
        feedforward_mult: int = 8,
        num_fusion_tokens: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        dino_repo: str = "facebookresearch/dinov2",
        dino_model_name: str = "dinov2_vitb14_reg",
        siglip_model_name: str = "google/siglip-base-patch16-224",
        text_model_name: str = "t5-base",
        freeze_backbones: bool = True,
        cache_language_embeddings: bool = True,
        max_text_length: int = 512,
        future_steps: int = 8,
        frame_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._last_train_step_start_time: Optional[float] = None
        self._cumulative_estimated_flops: float = 0.0
        self.future_steps = max(1, int(future_steps))
        self.frame_loss_weight = float(frame_loss_weight)

        dino_feature_dim = image_feature_dim // 2
        siglip_feature_dim = image_feature_dim - dino_feature_dim

        self.image_encoder = ImageEncoder(
            dino_repo=dino_repo,
            dino_model_name=dino_model_name,
            siglip_model_name=siglip_model_name,
            dino_feature_dim=dino_feature_dim,
            siglip_feature_dim=siglip_feature_dim,
            freeze_backbones=freeze_backbones,
        )
        self.language_encoder = LanguageEncoder(
            text_model_name=text_model_name,
            freeze_backbone=freeze_backbones,
            cache_embeddings=cache_language_embeddings,
            max_text_length=max_text_length,
        )

        self.image_proj = nn.Linear(image_feature_dim, embed_dim)
        self.language_proj = nn.Linear(language_feature_dim, embed_dim)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.modality_embeddings = nn.Parameter(torch.randn(3, embed_dim))
        self.fusion_tokens = nn.Parameter(torch.randn(1, num_fusion_tokens, embed_dim))

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * feedforward_mult,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pose_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, pose_dim),
        )
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, image_feature_dim),
        )
        self.future_pose_tokens = nn.Parameter(
            torch.randn(1, self.future_steps, embed_dim) * 0.02
        )
        self.criterion = nn.MSELoss()

    def _linear_flops(self, in_features: int, out_features: int, batch_tokens: int) -> int:
        return 2 * batch_tokens * in_features * out_features

    def _transformer_layer_flops(self, seq_len: int) -> int:
        embed_dim = int(self.hparams.embed_dim)
        ff_dim = embed_dim * int(self.hparams.feedforward_mult)

        attention_qkv = 3 * self._linear_flops(embed_dim, embed_dim, seq_len)
        attention_scores = 2 * int(self.hparams.num_heads) * seq_len * seq_len * (embed_dim // int(self.hparams.num_heads))
        attention_reduce = 2 * int(self.hparams.num_heads) * seq_len * seq_len * (embed_dim // int(self.hparams.num_heads))
        attention_out = self._linear_flops(embed_dim, embed_dim, seq_len)
        ffn = self._linear_flops(embed_dim, ff_dim, seq_len) + self._linear_flops(ff_dim, embed_dim, seq_len)
        return attention_qkv + attention_scores + attention_reduce + attention_out + ffn

    def estimate_forward_flops(self, batch_size: int) -> int:
        embed_dim = int(self.hparams.embed_dim)
        seq_len = int(self.hparams.num_fusion_tokens) + 3 + int(self.hparams.future_steps)
        image_proj = self._linear_flops(int(self.hparams.image_feature_dim), embed_dim, batch_size)
        language_proj = self._linear_flops(int(self.hparams.language_feature_dim), embed_dim, batch_size)
        pose_proj = (
            self._linear_flops(int(self.hparams.pose_dim), embed_dim, batch_size)
            + self._linear_flops(embed_dim, embed_dim, batch_size)
        )
        transformer = batch_size * int(self.hparams.num_layers) * self._transformer_layer_flops(seq_len)
        output_head = int(self.hparams.future_steps) * (
            self._linear_flops(embed_dim, embed_dim, batch_size)
            + self._linear_flops(embed_dim, embed_dim, batch_size)
            + self._linear_flops(embed_dim, int(self.hparams.pose_dim), batch_size)
        )
        return image_proj + language_proj + pose_proj + transformer + output_head

    def floating_point_ops(self, batch: Optional[dict[str, Any]] = None) -> int:
        if batch is None:
            batch_size = 1
        elif isinstance(batch, dict) and "image" in batch:
            batch_size = int(batch["image"].shape[0])
        else:
            batch_size = 1
        return self.estimate_forward_flops(batch_size)

    def forward(
        self,
        image: torch.Tensor,
        current_tool_pose: torch.Tensor,
        instruction: List[str],
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        image_embedding = self.image_proj(self.image_encoder(image))
        language_features = self.language_encoder(instruction, image.device)
        language_embedding = self.language_proj(language_features)
        pose_embedding = self.pose_proj(current_tool_pose)

        combined_input = torch.stack(
            [
                image_embedding + self.modality_embeddings[0],
                language_embedding + self.modality_embeddings[1],
                pose_embedding + self.modality_embeddings[2],
            ],
            dim=1,
        )
        fusion_tokens = self.fusion_tokens.expand(image.shape[0], -1, -1)
        future_tokens = self.future_pose_tokens.expand(image.shape[0], -1, -1)
        combined_input = torch.cat([fusion_tokens, combined_input, future_tokens], dim=1)

        # The future token block is decoded autoregressively: token k can use
        # the current observation and earlier future tokens, but not later
        # targets. Prefix modalities remain available to every future token.
        causal_mask = torch.triu(
            torch.ones(
                combined_input.shape[1],
                combined_input.shape[1],
                dtype=torch.bool,
                device=combined_input.device,
            ),
            diagonal=1,
        )
        transformer_output = self.transformer(combined_input, mask=causal_mask)
        future_latents = transformer_output[:, -self.future_steps :]
        predicted_poses = self.pose_head(future_latents)
        predicted_frame_features = self.frame_head(future_latents)
        if return_dict:
            return {
                "pose_latents": future_latents,
                "predicted_poses": predicted_poses,
                "predicted_frame_features": predicted_frame_features,
            }
        return predicted_poses

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        outputs = self(
            batch["image"],
            batch["current_tool_pose"],
            batch["instruction"],
            return_dict=True,
        )
        target_poses = batch.get("future_poses")
        if target_poses is None:
            target_poses = batch["target_pose"].unsqueeze(1)
        predicted_poses = outputs["predicted_poses"]
        if target_poses.ndim == 2:
            target_poses = target_poses.unsqueeze(1)
        pose_loss = self.criterion(predicted_poses, target_poses)

        frame_loss = predicted_poses.new_zeros(())
        if "future_images" in batch:
            future_images = batch["future_images"]
            batch_size, future_steps, channels, height, width = future_images.shape
            if future_steps != self.future_steps:
                raise ValueError(
                    f"Expected {self.future_steps} future images, got {future_steps}."
                )
            with torch.no_grad():
                target_features = self.image_encoder(
                    future_images.reshape(batch_size * future_steps, channels, height, width)
                ).reshape(batch_size, future_steps, -1)
            frame_loss = self.criterion(outputs["predicted_frame_features"], target_features)
        loss = pose_loss + self.frame_loss_weight * frame_loss
        is_train = stage == "train"
        self.log(
            f"{stage}_loss",
            loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=is_train,
            logger=True,
            batch_size=batch["image"].size(0),
        )
        self.log(
            f"{stage}_pose_loss",
            pose_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            batch_size=batch["image"].size(0),
        )
        self.log(
            f"{stage}_frame_loss",
            frame_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            batch_size=batch["image"].size(0),
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        self._last_train_step_start_time = time.perf_counter()
        loss = self._shared_step(batch, "train")

        batch_size = int(batch["image"].size(0))
        estimated_forward_flops = float(self.estimate_forward_flops(batch_size))
        estimated_train_step_flops = estimated_forward_flops * 3.0
        self._cumulative_estimated_flops += estimated_train_step_flops

        self.log(
            "train_estimated_fusion_flops",
            estimated_train_step_flops,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_cumulative_estimated_fusion_flops",
            self._cumulative_estimated_flops,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_batch_end(self, outputs, batch: dict, batch_idx: int) -> None:
        if self._last_train_step_start_time is None:
            return

        elapsed = max(time.perf_counter() - self._last_train_step_start_time, 1e-6)
        batch_size = int(batch["image"].size(0))
        self.log(
            "train_step_time",
            elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_samples_per_sec",
            batch_size / elapsed,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_name = self.hparams.lr_scheduler.lower()
        if scheduler_name == "constant":
            return optimizer

        if scheduler_name == "cosine":
            scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer,
                num_warmup_steps=self.hparams.lr_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        raise ValueError(
            f"Unsupported lr_scheduler '{self.hparams.lr_scheduler}'. "
            "Expected 'constant' or 'cosine'."
        )


if __name__ == "__main__":
    model = Stage1Transformer()
    image = torch.randn(2, 3, 224, 224)
    tool_pose = torch.randn(2, 16)
    instruction = ["pick up the tool", "place the block on the red square"]
    predicted_pose = model(image, tool_pose, instruction)
    print("Predicted Pose shape:", predicted_pose.shape)
