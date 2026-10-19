import argparse
import os
from train.train import train

from accelerate.logging import get_logger


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train the P2P-T Stage 2 action expert.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/ur10e_p2p_t.yaml",
        help="Path to the P2P-T configuration file.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Enable DeepSpeed and pass the path to its config file or an already initialized DeepSpeed config dictionary",
    )
    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default=None,
        help="Pretrained text encoder name or path if not the same as model_name",
    )
    parser.add_argument(
        "--pretrained_vision_encoder_name_or_path",
        type=str,
        default=None,
        help="Pretrained vision encoder name or path if not the same as model_name",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--load_from_hdf5",
        action="store_true",
        default=False,
        help="Load the P2P-T UR10e episodes directly from HDF5 files.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=8, help="Batch size (per device) for the sampling dataloader."
    )
    parser.add_argument(
        "--num_sample_batches", type=int, default=2, help="Number of batches to sample from the dataset."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_period",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_period`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Path or name of a pretrained RDT backbone or P2P-T checkpoint to load.\n"
            "   This can be either:\n"
            "   - a string, the model id of a published RDT backbone or P2P-T checkpoint,\n"
            "   - a path to a *directory* containing model weights saved using [`~RDTRunner.save_pretrained`] method, e.g., `./my_model_directory/`.\n"
            "   - a path to model checkpoint (*.pt), .e.g, `my_model_directory/checkpoint-10000/pytorch_model/mp_rank_00_model_states.pt`"
            "   - `None` if you are randomly initializing model using configuration at `config_path`."
        )
    )
    parser.add_argument(
        "--use_stage1_condition",
        action="store_true",
        default=False,
        help="Use the frozen Stage 1 world model as a pose-latent condition.",
    )
    parser.add_argument(
        "--stage1_config_path",
        type=str,
        default="stage1/config/stage1.yaml",
        help="Path to the Stage 1 YAML config.",
    )
    parser.add_argument(
        "--stage1_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained Stage 1 checkpoint file.",
    )
    parser.add_argument(
        "--stage1_checkpoint_dir",
        type=str,
        default=None,
        help="Directory to search for the Stage 1 checkpoint when `stage1_checkpoint_path` is not set.",
    )
    parser.add_argument(
        "--stage1_checkpoint_name",
        type=str,
        default="last.ckpt",
        help="Filename to search for inside `stage1_checkpoint_dir`.",
    )
    parser.add_argument(
        "--stage1_target_camera_index",
        type=int,
        default=0,
        help="Camera index from the latest observation frame to feed into Stage 1.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cond_mask_prob",
        type=float,
        default=0.1,
        help=(
            "The probability to randomly mask the conditions (except states) during training. "
            "If set to 0, the conditions are not masked."
        ),
    )
    parser.add_argument(
        "--cam_ext_mask_prob",
        type=float,
        default=-1.0,
        help=(
            "The probability to randomly mask the external camera image during training. "
            "If set to < 0, the external camera image is masked with the probability of `cond_mask_prob`."
        ),
    )
    parser.add_argument(
        "--state_noise_snr",
        type=float,
        default=None,
        help=(
            "The signal-to-noise ratio (SNR, unit: dB) for adding noise to the states. "
            "Default is None, which means no noise is added."
        ),
    )
    parser.add_argument(
        "--image_aug",
        action="store_true",
        default=False,
        help="Whether or not to apply image augmentation (ColorJitter, blur, noise, etc) to the input images.",
    )
    parser.add_argument(
        "--precomp_lang_embed",
        action="store_true",
        default=False,
        help="Whether or not to use precomputed language embeddings.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="The moving average coefficient for each dataset's loss.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--sample_period",
        type=int,
        default=-1,
        help=(
            "Run sampling every X steps. During the sampling phase, the model will sample a trajectory"
            " and report the error between the sampled trajectory and groud-truth trajectory"
            " in the training batch."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument('--dataset_type',
        type=str, 
        default="p2p_t",
        required=False,
        help="Dataset contract to load. The release ships `p2p_t`."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train(args, logger)
