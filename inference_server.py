import argparse
import os
import pickle
import socket
import struct

import cv2
import numpy as np
import torch
import yaml
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_model

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from models.stage1_conditioned_rdt_runner import Stage1ConditionedRDTRunner

UR10E_ARM_DOF = 6
UR10E_DEPLOY_ACTION_DIM = 7
UR10E_LEGACY_ACTION_DIM = 8
UR10E_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def recv_exact(conn, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError("Socket closed while receiving payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_image(encoded_or_array):
    if encoded_or_array is None:
        return None
    if isinstance(encoded_or_array, bytes):
        encoded_or_array = np.frombuffer(encoded_or_array, dtype=np.uint8)
    if isinstance(encoded_or_array, np.ndarray) and encoded_or_array.ndim == 1:
        image = cv2.imdecode(encoded_or_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(encoded_or_array, np.ndarray):
        image = encoded_or_array
        if image.ndim == 3 and image.shape[-1] == 3:
            return image
    raise ValueError(f"Unsupported image payload type/shape: {type(encoded_or_array)}")


def as_float_array(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"`{name}` is empty.")
    return array


def format_ur10e_qpos_for_state(qpos):
    """Canonicalize UR10e qpos as [6 UR joints in RTDE/ROS-control order, gripper].

    Joint order is:
    shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.
    Deployment observations are 7D. Older data/checkpoints may expose an 8D
    vector with an unused compatibility column before gripper; keep the first
    six arm joints and the final gripper value.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[-1] == UR10E_DEPLOY_ACTION_DIM:
        return qpos
    if qpos.shape[-1] == UR10E_LEGACY_ACTION_DIM:
        return np.concatenate([qpos[..., :UR10E_ARM_DOF], qpos[..., -1:]], axis=-1)
    if qpos.shape[-1] == UR10E_ARM_DOF:
        gripper = np.zeros((*qpos.shape[:-1], 1), dtype=np.float32)
        return np.concatenate([qpos, gripper], axis=-1)
    raise ValueError(f"Unsupported UR10e qpos shape {qpos.shape}")


def format_ur10e_action_for_deployment(action):
    """Return [6 UR joint targets in RTDE/ROS-control order, gripper].

    This mirrors Trustworthy_Tool_Manip/inference_server.py: first six values
    are joint positions and the final value is gripper state. If an old 8D
    vector appears, the middle blank/compatibility value is ignored by taking
    the final element as gripper.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] < UR10E_DEPLOY_ACTION_DIM:
        raise ValueError(f"Expected at least 7 UR10e action values, got shape {action.shape}")
    return np.concatenate([action[..., :UR10E_ARM_DOF], action[..., -1:]], axis=-1)


class RDTInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, args.dtype)
        with open(args.config_path, "r") as fp:
            self.config = yaml.safe_load(fp)

        self.img_history_size = self.config["common"]["img_history_size"]
        self.num_cameras = self.config["common"]["num_cameras"]
        self.action_chunk_size = self.config["common"]["action_chunk_size"]
        self.control_frequency = args.ctrl_freq
        self.arm_dim = args.arm_dim
        self.gripper_dim = args.gripper_dim
        if args.robot != "ur10e":
            raise ValueError("The P2P-T release server supports only the UR10e embodiment.")
        self.arm_dim = UR10E_ARM_DOF
        self.gripper_dim = 1
        self.qpos_dim = self.arm_dim + self.gripper_dim
        self.state_indices = self._resolve_state_indices(args.robot)

        self.image_processor, self.vision_model = self._build_vision_encoder()
        self.policy = self._build_policy()
        self.lang_tokens = self._load_or_encode_language(args)
        self.lang_attn_mask = torch.ones(
            self.lang_tokens.shape[:2], dtype=torch.bool, device=self.device
        )

        self.reset()
        print("--- RDT Inference Server: warming up ---")
        if args.robot == "ur10e":
            print("UR10e joint order:", ", ".join(UR10E_JOINT_ORDER))
            print("UR10e action format: [first 6 dims = joint positions, last dim = gripper]")
        self.warmup()

    def _resolve_state_indices(self, robot):
        if robot == "ur10e":
            return [
                STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(UR10E_ARM_DOF)
            ] + [STATE_VEC_IDX_MAPPING["gripper_open"]]
        raise ValueError(f"Unsupported robot layout: {robot}")

    def _build_vision_encoder(self):
        vision_encoder = SiglipVisionTower(
            vision_tower=self.args.pretrained_vision_encoder_name_or_path,
            args=None,
        )
        return vision_encoder.image_processor, vision_encoder

    def _runner_kwargs(self):
        img_cond_len = (
            self.img_history_size
            * self.num_cameras
            * self.vision_model.num_patches
        )
        return dict(
            action_dim=self.config["common"]["state_dim"],
            pred_horizon=self.action_chunk_size,
            config=self.config["model"],
            lang_token_dim=self.config["model"]["lang_token_dim"],
            img_token_dim=self.config["model"]["img_token_dim"],
            state_token_dim=self.config["model"]["state_token_dim"],
            max_lang_cond_len=self.config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                (
                    "image",
                    (
                        self.img_history_size,
                        self.num_cameras,
                        -self.vision_model.num_patches,
                    ),
                ),
            ],
            lang_pos_embed_config=[
                ("lang", -self.config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.dtype,
        )

    def _build_policy(self):
        stage1_cfg = self.config.get("stage1", {})
        use_stage1 = bool(stage1_cfg.get("enabled", False))
        runner_cls = Stage1ConditionedRDTRunner if use_stage1 else RDTRunner
        runner_kwargs = self._runner_kwargs()
        if use_stage1:
            runner_kwargs.update(
                stage1_config_path=stage1_cfg.get("config_path", "stage1/config/stage1.yaml"),
                stage1_checkpoint_path=stage1_cfg.get("checkpoint_path"),
                stage1_checkpoint_dir=stage1_cfg.get("checkpoint_dir"),
                stage1_checkpoint_name=stage1_cfg.get("checkpoint_name", "last.ckpt"),
                stage1_target_camera_index=stage1_cfg.get("target_camera_index", 0),
            )

        pretrained = self.args.pretrained_model_name_or_path
        self._align_runner_kwargs_with_checkpoint(pretrained, runner_kwargs, use_stage1)
        if pretrained and not os.path.isfile(pretrained):
            policy = runner_cls.from_pretrained(pretrained, **runner_kwargs)
        else:
            policy = runner_cls(**runner_kwargs)
            if pretrained:
                self._load_checkpoint(policy, pretrained)
        return policy

    def _align_runner_kwargs_with_checkpoint(self, pretrained, runner_kwargs, use_stage1):
        checkpoint_path = self._resolve_local_weight_file(pretrained)
        if checkpoint_path is None:
            return

        x_pos_shape = self._read_checkpoint_tensor_shape(checkpoint_path, "model.x_pos_embed")
        if x_pos_shape is None or len(x_pos_shape) != 3:
            return

        num_state_tokens = self._runner_state_token_count(use_stage1)
        pred_horizon = int(x_pos_shape[1]) - num_state_tokens - 2
        if pred_horizon <= 0:
            return
        if pred_horizon == int(runner_kwargs["pred_horizon"]):
            return

        print(
            "Checkpoint pred_horizon differs from config: "
            f"using {pred_horizon} from {checkpoint_path} "
            f"instead of {runner_kwargs['pred_horizon']}."
        )
        runner_kwargs["pred_horizon"] = pred_horizon
        self.action_chunk_size = pred_horizon
        self.config["common"]["action_chunk_size"] = pred_horizon

    def _runner_state_token_count(self, use_stage1):
        if not use_stage1:
            return 1
        stage1_config = self.config.get("stage1", {})
        config_path = stage1_config.get("config_path", "stage1/config/stage1.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                stage1_yaml = yaml.safe_load(handle) or {}
            model_config = stage1_yaml.get("model", {})
            data_config = stage1_yaml.get("data", {})
            future_steps = model_config.get(
                "future_steps",
                data_config.get(
                    "future_steps", data_config.get("recipe", {}).get("future_steps", 1)
                ),
            )
            return 1 + max(1, int(future_steps))
        except (OSError, TypeError, ValueError):
            # Keep compatibility with the original one-pose-token wrapper when
            # loading a legacy deployment config.
            return 2

    @staticmethod
    def _resolve_local_weight_file(pretrained):
        if not pretrained:
            return None
        if os.path.isfile(pretrained):
            return pretrained
        if not os.path.isdir(pretrained):
            return None

        safetensors_path = os.path.join(pretrained, SAFETENSORS_SINGLE_FILE)
        if os.path.isfile(safetensors_path):
            return safetensors_path

        pickle_path = os.path.join(pretrained, PYTORCH_WEIGHTS_NAME)
        if os.path.isfile(pickle_path):
            return pickle_path
        return None

    @staticmethod
    def _read_checkpoint_tensor_shape(checkpoint_path, key):
        if checkpoint_path.endswith(".safetensors"):
            with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
                if key not in handle.keys():
                    return None
                return tuple(handle.get_tensor(key).shape)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("module", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        tensor = state_dict.get(key) if isinstance(state_dict, dict) else None
        if tensor is None:
            return None
        return tuple(tensor.shape)

    def _load_checkpoint(self, policy, checkpoint_path):
        print(f"Loading RDT weights from {checkpoint_path}")
        if checkpoint_path.endswith(".safetensors"):
            load_model(policy, checkpoint_path, strict=False)
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("module", checkpoint)
        model_state = policy.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        policy.load_state_dict(compatible_state, strict=False)

    def _load_or_encode_language(self, args):
        if args.lang_embeddings_path:
            lang = torch.load(args.lang_embeddings_path, map_location="cpu")
            if isinstance(lang, dict):
                lang = lang.get("embeddings", lang.get("lang_embeds", lang))
            lang = torch.as_tensor(lang)
            if lang.ndim == 2:
                lang = lang.unsqueeze(0)
            return lang.to(self.device, dtype=self.dtype)

        if not args.instruction:
            raise ValueError("Provide either --lang_embeddings_path or --instruction.")

        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=self.config["dataset"]["tokenizer_max_length"],
            device=self.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        tokens = tokenizer(
            args.instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )["input_ids"].to(self.device)
        text_encoder = text_encoder.to(self.device, dtype=self.dtype).eval()
        with torch.no_grad():
            return text_encoder(tokens).last_hidden_state.detach().to(dtype=self.dtype)

    def reset(self):
        self.policy.eval()
        self.vision_model.eval()
        self.policy = self.policy.to(self.device, dtype=self.dtype)
        self.vision_model = self.vision_model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def warmup(self):
        h = self.image_processor.size["height"]
        w = self.image_processor.size["width"]
        images = [Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))] * (
            self.img_history_size * self.num_cameras
        )
        qpos = np.zeros((self.qpos_dim,), dtype=np.float32)
        warmup_obs = {"images": images, "qpos": qpos}
        if self.config.get("stage1", {}).get("enabled", False):
            warmup_obs["current_tool_pose"] = np.eye(4, dtype=np.float32).reshape(-1)
            warmup_obs["instruction"] = self.args.instruction or "warm up"
        for _ in range(2):
            _ = self.infer(warmup_obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _format_qpos_to_state(self, qpos):
        if self.args.robot == "ur10e":
            qpos = format_ur10e_qpos_for_state(qpos)
        qpos = torch.as_tensor(qpos, device=self.device, dtype=self.dtype).reshape(1, 1, -1)
        if qpos.shape[-1] != len(self.state_indices):
            raise ValueError(
                f"Expected qpos dim {len(self.state_indices)}, got {qpos.shape[-1]}."
            )
        state = torch.zeros(
            (1, 1, self.config["model"]["state_token_dim"]),
            device=self.device,
            dtype=self.dtype,
        )
        state[:, :, self.state_indices] = qpos
        state_elem_mask = torch.zeros(
            (1, self.config["model"]["state_token_dim"]),
            device=self.device,
            dtype=self.dtype,
        )
        state_elem_mask[:, self.state_indices] = 1
        return state, state_elem_mask

    def _unformat_action(self, trajectory):
        action = trajectory[:, :, self.state_indices].to(torch.float32).cpu().numpy()[0]
        if self.args.robot == "ur10e":
            action = format_ur10e_action_for_deployment(action)
        if self.args.return_chunk:
            return action
        return action[0]

    def _prepare_images(self, images):
        if len(images) == 0:
            raise ValueError("No images provided.")
        target_len = self.img_history_size * self.num_cameras
        if len(images) < target_len:
            images = [images[0]] * (target_len - len(images)) + list(images)
        images = list(images)[-target_len:]

        processed = []
        stage1_image = None
        stage1_image_index = (
            (self.img_history_size - 1) * self.num_cameras
            + self.config.get("stage1", {}).get("target_camera_index", 0)
        )
        background_color = np.array(
            [int(x * 255) for x in self.image_processor.image_mean],
            dtype=np.uint8,
        ).reshape(1, 1, 3)
        background = np.ones(
            (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
                3,
            ),
            dtype=np.uint8,
        ) * background_color

        for idx, image in enumerate(images):
            if image is None:
                pil_image = Image.fromarray(background)
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")

            if self.config["dataset"].get("image_aspect_ratio", "pad") == "pad":
                pil_image = self._expand_to_square(
                    pil_image,
                    tuple(int(x * 255) for x in self.image_processor.image_mean),
                )
            if idx == stage1_image_index:
                stage1_image = pil_image.resize((224, 224))
            processed.append(
                self.image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"][0]
            )

        image_tensor = torch.stack(processed, dim=0).to(self.device, dtype=self.dtype)
        return image_tensor, stage1_image

    @staticmethod
    def _expand_to_square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        if width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

    @torch.no_grad()
    def infer(self, obs):
        image_tensor, stage1_pil_image = self._prepare_images(obs["images"])
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)
        states, state_elem_mask = self._format_qpos_to_state(obs["qpos"])
        ctrl_freqs = torch.tensor([self.control_frequency], device=self.device)

        predict_kwargs = dict(
            lang_tokens=self.lang_tokens,
            lang_attn_mask=self.lang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
        )
        if self.config.get("stage1", {}).get("enabled", False):
            if "current_tool_pose" not in obs:
                raise ValueError("Stage1-conditioned inference requires `current_tool_pose`.")
            if "instruction" not in obs and not self.args.instruction:
                raise ValueError("Stage1-conditioned inference requires raw instruction text.")
            stage1_image_np = np.asarray(stage1_pil_image, dtype=np.float32) / 255.0
            stage1_image = torch.from_numpy(stage1_image_np).permute(2, 0, 1).unsqueeze(0)
            predict_kwargs.update(
                stage1_images=stage1_image.to(self.device, dtype=torch.float32),
                current_tool_poses=torch.as_tensor(
                    obs["current_tool_pose"], device=self.device, dtype=torch.float32
                ).reshape(1, -1),
                instructions=[obs.get("instruction", self.args.instruction)],
            )

        trajectory = self.policy.predict_action(**predict_kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return self._unformat_action(trajectory)


def build_observation(data):
    images = []
    if "img_history" in data:
        images = [decode_image(item) for item in data["img_history"]]
    elif "images" in data:
        raw_images = data["images"]
        if isinstance(raw_images, dict):
            for history_idx in range(max(len(v) for v in raw_images.values())):
                for key in sorted(raw_images):
                    values = raw_images[key]
                    if history_idx < len(values):
                        images.append(decode_image(values[history_idx]))
        else:
            images = [decode_image(item) for item in raw_images]

    if "qpos_history" in data:
        qpos = as_float_array(data["qpos_history"], "qpos_history")[-1]
    elif "qpos" in data:
        qpos = as_float_array(data["qpos"], "qpos")
    elif "arm_history" in data and "gripper_history" in data:
        arm = as_float_array(data["arm_history"], "arm_history")[-1].reshape(-1)
        gripper = as_float_array(data["gripper_history"], "gripper_history")[-1].reshape(-1)
        qpos = np.concatenate([arm, gripper], axis=0)
    else:
        arm = as_float_array(data.get("arm"), "arm").reshape(-1)
        gripper = as_float_array(data.get("gripper"), "gripper").reshape(-1)
        qpos = np.concatenate([arm, gripper], axis=0)

    obs = {"images": images, "qpos": qpos}
    for key in ["instruction", "current_tool_pose", "target_pose", "pose"]:
        if key in data:
            obs["current_tool_pose" if key == "pose" else key] = data[key]
    return obs


def split_action(action, robot, arm_dim, gripper_dim):
    action = np.asarray(action, dtype=np.float32)
    if robot == "ur10e":
        action = format_ur10e_action_for_deployment(action)
        return action[..., :UR10E_ARM_DOF], action[..., -1:]
    arm = action[..., :arm_dim]
    gripper = action[..., arm_dim:arm_dim + gripper_dim]
    return arm, gripper


def main():
    parser = argparse.ArgumentParser(description="P2P-T UR10e socket inference server.")
    parser.add_argument("--config_path", default="configs/ur10e_p2p_t.yaml")
    parser.add_argument("--pretrained_model_name_or_path", "--ckpt", dest="pretrained_model_name_or_path", required=True)
    parser.add_argument("--pretrained_vision_encoder_name_or_path", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--pretrained_text_encoder_name_or_path", default="google/t5-v1_1-xxl")
    parser.add_argument("--lang_embeddings_path", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--ctrl_freq", type=int, default=30)
    parser.add_argument("--robot", choices=["ur10e"], default="ur10e")
    parser.add_argument("--arm_dim", type=int, default=6)
    parser.add_argument("--gripper_dim", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--return_chunk", action="store_true")
    args = parser.parse_args()

    engine = RDTInferenceEngine(args)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"RDT server online. Listening on {args.host}:{args.port}")

    header_struct = struct.Struct("Q")
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Inference client connected: {addr}")
        try:
            while True:
                header = recv_exact(conn, header_struct.size)
                msg_size = header_struct.unpack(header)[0]
                data = pickle.loads(recv_exact(conn, msg_size))
                obs = build_observation(data)
                action = engine.infer(obs)
                arm_action, gripper_action = split_action(
                    action,
                    robot=args.robot,
                    arm_dim=args.arm_dim,
                    gripper_dim=args.gripper_dim,
                )
                response = {
                    "action": np.asarray(action, dtype=np.float32),
                    "arm": np.asarray(arm_action, dtype=np.float32),
                    "gripper": np.asarray(gripper_action, dtype=np.float32),
                }
                payload = pickle.dumps(response)
                conn.sendall(header_struct.pack(len(payload)) + payload)
        except (ConnectionError, EOFError):
            print(f"Inference client disconnected: {addr}")
        except Exception as exc:
            print(f"Inference error for {addr}: {exc}")
            try:
                payload = pickle.dumps({"error": str(exc)})
                conn.sendall(header_struct.pack(len(payload)) + payload)
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
