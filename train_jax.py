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
from jax_smi import initialise_tracking

initialise_tracking()

model_config = ModelConfig.small(
    model_type="seqcond", num_thetas=2, seqcond_heads=24, maxlen=1024, vocab_size=100300
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
        "--fsdp",
        action="store_true",
        help="Enable fully sharded data parallel (FSDP) training",
    )
    parser.add_argument(
        "--freeze-thetas",
        action="store_true",
        help="If set, keep theta parameters frozen (not trainable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size (int)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
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
        default=None,
        help="Log every n steps",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override wandb project",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=None,
        help="Number of batches to prefetch to device (0 disables)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides base_lr)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["seqcond", "transformer"],
        default=None,
        help="Model type: seqcond or transformer",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Model dimension (d_model)",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help="Feed-forward dimension (d_ff)",
    )
    parser.add_argument(
        "--num-thetas",
        type=int,
        default=None,
        help="Number of theta parameters for seqcond",
    )
    parser.add_argument(
        "--derivative",
        type=int,
        default=0,
        help="Derivative order for seqcond (0, 1, or 2)",
    )
    parser.add_argument(
        "--anchor",
        type=int,
        default=0,
        help="Number of anchor heads for seqcond",
    )
    parser.add_argument(
        "--seqcond-heads",
        type=int,
        default=None,
        help="Number of seqcond heads",
    )
    parser.add_argument(
        "--generate-every-n-steps",
        type=int,
        default=None,
        help="Generate samples every n steps",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_multiple_tpus:
        config.training.use_multiple_tpus = True
    if args.fsdp:
        config.training.full_shard_data_parallel = True
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
    if args.prefetch_batches is not None:
        config.training.prefetch_batches = max(0, int(args.prefetch_batches))
    if args.lr is not None:
        config.training.base_lr = float(args.lr)
    if args.model is not None:
        config.model.model_type = args.model
    if args.num_layers is not None:
        config.model.num_layers = int(args.num_layers)
    if args.d_model is not None:
        config.model.d_model = int(args.d_model)
    if args.d_ff is not None:
        config.model.d_ff = int(args.d_ff)
    if args.num_thetas is not None:
        config.model.num_thetas = int(args.num_thetas)
    config.model.derivative_order = int(args.derivative)
    config.model.num_anchor_heads = int(args.anchor)
    if args.seqcond_heads is not None:
        config.model.seqcond_heads = int(args.seqcond_heads)
    if args.generate_every_n_steps is not None:
        config.training.generate_every_n_steps = int(args.generate_every_n_steps)

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
