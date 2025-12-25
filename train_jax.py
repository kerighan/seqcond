"""
JAX/Flax training script for language models.
Uses the high-level training API from lib.jax.train.
"""

import os
import argparse
import logging
from dataclasses import fields

# 1. Environment and Distributed Init MUST happen first
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "tpu"

logging.getLogger("jax").setLevel(logging.ERROR)

# Pre-define parse_args so we can use it before full imports
def parse_args():
    parser = argparse.ArgumentParser(description="Train SeqCond model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # # Distributed / Debugging
    # parser.add_argument("--jax-distributed", action="store_true", help="Initialize JAX distributed system (for multi-host TPU)")
    # parser.add_argument("--jax-smi", action="store_true", help="Enable JAX SMI tracking (memory profiler)")

    # Base Configuration
    parser.add_argument("--size", choices=["small", "medium", "large", "xlarge"], default="small", help="Base model size configuration")
    parser.add_argument("--resume-step", type=int, default=None, help="Step to resume training from")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Specific checkpoint file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model Configuration Overrides
    grp_model = parser.add_argument_group("Model Overrides")
    grp_model.add_argument("--model-type", choices=["seqcond", "transformer"], default=None, help="Override model architecture type")
    grp_model.add_argument("--num-layers", type=int, default=None, help="Override number of layers")
    grp_model.add_argument("--d-model", type=int, default=None, help="Override model dimension")
    grp_model.add_argument("--d-ff", type=int, default=None, help="Override feed-forward dimension")
    grp_model.add_argument("--num-thetas", type=int, default=None, help="Override number of theta parameters")
    grp_model.add_argument("--ratio", type=int, default=None, dest="seqcond_ratio", help="Override seqcond to transformer ratio")
    grp_model.add_argument("--derivative", type=int, default=None, dest="derivative_order", help="Override derivative order (0, 1, 2)")
    grp_model.add_argument("--anchor", type=int, default=None, dest="num_anchor_heads", help="Override number of anchor heads")
    grp_model.add_argument("--seqcond-heads", type=int, default=None, help="Override number of seqcond heads")
    grp_model.add_argument("--expand", type=float, default=2.0, dest="expand_factor", help="Override expand factor")
    grp_model.add_argument("--maxlen", type=int, default=1024, help="Context length (affects model and training)")

    # Training Configuration Overrides
    grp_train = parser.add_argument_group("Training Overrides")
    grp_train.add_argument("--use-multiple-tpus", action="store_true", help="Enable pmap/multi-device training")
    grp_train.add_argument("--fsdp", action="store_true", dest="full_shard_data_parallel", help="Enable FSDP")
    grp_train.add_argument("--freeze-thetas", action="store_true", help="Freeze theta parameters")
    grp_train.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    grp_train.add_argument("--total-steps", type=int, default=None, help="Override total training steps")
    grp_train.add_argument("--save-every-n-steps", type=int, default=None, help="Checkpoint interval")
    grp_train.add_argument("--log-every-n-steps", type=int, default=None, help="Logging interval")
    grp_train.add_argument("--generate-every-n_steps", type=int, default=None, help="Generation sample interval")
    grp_train.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    grp_train.add_argument("--prefetch-batches", type=int, default=None, help="Number of batches to prefetch")
    grp_train.add_argument("--lr", type=float, default=None, dest="base_lr", help="Override learning rate")
    grp_train.add_argument("--no-remat", action="store_true", help="Disable gradient checkpointing (rematerialization)")

    return parser.parse_args()

# Parse args immediately to configure environment
args = parse_args()

import jax

jax.distributed.initialize()

# Now import application modules
from seqcond.config import Config, ModelConfig, TrainingConfig
from seqcond.dataset import tokenizer
from seqcond.jax import train
from jax_smi import initialise_tracking


def get_config(args) -> Config:
    """Builds the configuration object from arguments."""
    
    # 1. Base Model Config
    model_factory = getattr(ModelConfig, args.size)
    
    # Collect overrides for ModelConfig
    model_overrides = {}
    
    # Direct mappings (name matches)
    for field in ["num_layers", "d_model", "d_ff", "num_thetas", "seqcond_heads", "model_type", "seqcond_ratio", "expand_factor"]:
        val = getattr(args, field)
        if val is not None:
            model_overrides[field] = val

    # Renamed/Special mappings
    if args.derivative_order is not None: model_overrides["derivative_order"] = args.derivative_order
    if args.num_anchor_heads is not None: model_overrides["num_anchor_heads"] = args.num_anchor_heads
    if args.maxlen is not None: model_overrides["maxlen"] = args.maxlen

    # Create model config
    model_config = model_factory(**model_overrides)
    
    # Remat logic
    if args.no_remat:
        model_config.remat = False

    # 2. Training Config
    # Default values suitable for this script (preserved from original file)
    training_defaults = dict(
        batch_size=1,
        maxlen=args.maxlen, # Sync with model maxlen
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
    )

    training_config = TrainingConfig(**training_defaults)

    # Apply Overrides
    if args.use_multiple_tpus: training_config.use_multiple_tpus = True
    if args.full_shard_data_parallel: training_config.full_shard_data_parallel = True
    if args.freeze_thetas: training_config.train_thetas = False
    
    # Map other training args if they are not None
    train_fields = [
        "batch_size", "total_steps", "save_every_n_steps", "log_every_n_steps", 
        "wandb_project", "prefetch_batches", "base_lr", "generate_every_n_steps"
    ]
    for field in train_fields:
        val = getattr(args, field)
        if val is not None:
            setattr(training_config, field, val)

    return Config(model=model_config, training=training_config)


def main():
    initialise_tracking()
    print("JAX SMI tracking enabled")
    
    config = get_config(args)

    # Resume Logic
    resume_ckpt = args.resume_checkpoint
    if resume_ckpt is None and args.resume_step is not None:
        ckpt_name = f"{config.name}_step{args.resume_step}.pkl"
        candidate = os.path.join(config.training.checkpoint_dir, ckpt_name)
        if os.path.exists(candidate):
            resume_ckpt = candidate
            print(f"Resuming from {resume_ckpt}")
        else:
            print(f"Checkpoint for step {args.resume_step} not found at {candidate}")

    # Start Training
    train(
        config=config,
        tokenizer=tokenizer,
        seed=args.seed,
        resume_checkpoint=resume_ckpt,
    )


if __name__ == "__main__":
    main()
