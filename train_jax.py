"""
JAX/Flax training script for language models.
Uses the high-level training API from lib.jax.train.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "tpu"

import argparse
import logging

logging.getLogger("jax").setLevel(logging.ERROR)

from seqcond.config import Config, ModelConfig, TrainingConfig
from seqcond.dataset import tokenizer
from seqcond.jax import train


model_config = ModelConfig.small(
    model_type="seqcond", num_thetas=4, seqcond_heads=16, maxlen=1024
)
config = Config(
    model=model_config,
    training=TrainingConfig(
        batch_size=1,
        maxlen=1024,
        base_lr=1e-3,
        warmup_steps=2000,
        total_steps=500000,
        weight_decay=1e-2,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        grad_accum_steps=1,
        mixed_precision="bfloat16",
        keep_weights_fp32=True,
        log_every_n_steps=1000,
        generate_every_n_steps=10000,
        save_every_n_steps=50000,
        checkpoint_dir="checkpoints",
        use_wandb=True,
        wandb_project="slm-training-len768",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-multiple-tpus",
        action="store_true",
        help="Enable pmap data-parallel training across all TPU devices",
    )
    parser.add_argument(
        "--freeze-thetas",
        action="store_true",
        help="If set, keep theta parameters frozen (not trainable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Override training batch size (int)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=500000,
        help="Override total number of training steps (int)",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=None,
        help="Override checkpoint save interval (int)",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        help="Log every n steps"
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override wandb project",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_multiple_tpus:
        config.training.use_multiple_tpus = True
    if args.freeze_thetas:
        config.training.train_thetas = False
    if args.batch_size is not None:
        config.training.batch_size = int(args.batch_size)
    if args.total_steps is not None:
        config.training.total_steps = int(args.total_steps)
    if args.save_every_n_steps is not None:
        config.training.save_every_n_steps = int(args.save_every_n_steps)
    if args.wandb_project is not None:
        config.training.wandb_project = args.wandb_project
    if args.log_every_n_steps is not None:
        config.training.log_every_n_steps = args.log_every_n_steps

    resume_ckpt = args.resume_checkpoint
    if resume_ckpt is None and args.resume_step is not None:
        ckpt_name = f"{config.name}_step{args.resume_step}.pkl"
        candidate = os.path.join(config.training.checkpoint_dir, ckpt_name)
        if os.path.exists(candidate):
            resume_ckpt = candidate
            print(f"Resuming from {resume_ckpt}")
        else:
            print(f"Checkpoint for step {args.resume_step} not found at {candidate}")

    train(
        config=config,
        tokenizer=tokenizer,
        seed=args.seed,
        resume_checkpoint=resume_ckpt,
    )
