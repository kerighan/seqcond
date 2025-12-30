"""
Script to count parameters of the SeqCond model without training.
Uses the same argument parsing as train_jax.py.
"""

import os
import argparse
import logging
from dataclasses import fields

# 1. Environment and Distributed Init MUST happen first
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["JAX_PLATFORMS"] = "tpu" # We don't strictly need TPU for counting params, CPU is fine usually, but let's keep it consistent if needed or let JAX decide.
# Actually, for just counting params, we might want to avoid TPU init if possible to be faster/lighter, but let's stick to the structure.
# Commenting out JAX_PLATFORMS to allow running on CPU if TPU is not available/needed for this lightweight task.

logging.getLogger("jax").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Count SeqCond model parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base Configuration
    parser.add_argument("--size", choices=["small", "medium", "large", "xlarge"], default="small", help="Base model size configuration")
    parser.add_argument("--resume-step", type=int, default=None, help="Step to resume training from (Ignored)")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Specific checkpoint file path (Ignored)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model Configuration Overrides
    grp_model = parser.add_argument_group("Model Overrides")
    grp_model.add_argument("--model-type", choices=["seqcond", "transformer", "bivector"], default=None, help="Override model architecture type")
    grp_model.add_argument("--num-layers", type=int, default=None, help="Override number of layers")
    grp_model.add_argument("--d-model", type=int, default=None, help="Override model dimension")
    grp_model.add_argument("--d-ff", type=int, default=None, help="Override feed-forward dimension")
    grp_model.add_argument("--num-thetas", type=int, default=None, help="Override number of theta parameters")
    grp_model.add_argument("--ratio", type=int, default=None, dest="seqcond_ratio", help="Override seqcond to transformer ratio")
    grp_model.add_argument("--derivative", type=int, default=None, dest="derivative_order", help="Override derivative order (0, 1, 2)")
    grp_model.add_argument("--anchor", type=int, default=None, dest="num_anchor_heads", help="Override number of anchor heads")
    grp_model.add_argument("--seqcond-heads", type=int, default=None, help="Override number of seqcond heads")
    grp_model.add_argument("--seqcond-query-heads", type=int, default=6, dest="num_query_heads", help="Override number of seqcond query heads")
    grp_model.add_argument("--expand", type=float, default=1.0, dest="expand_factor", help="Override expand factor")
    grp_model.add_argument("--maxlen", type=int, default=1024, help="Context length (affects model and training)")
    grp_model.add_argument("--chunk", type=int, default=0, dest="chunk_size", help="Chunk size for cumsum (0 = normal)")

    # Training Configuration Overrides (kept for compatibility, but largely ignored)
    grp_train = parser.add_argument_group("Training Overrides")
    grp_train.add_argument("--use-multiple-tpus", action="store_true", help="Enable pmap/multi-device training (Ignored)")
    grp_train.add_argument("--fsdp", action="store_true", dest="full_shard_data_parallel", help="Enable FSDP (Ignored)")
    grp_train.add_argument("--freeze-thetas", action="store_true", help="Freeze theta parameters (Ignored)")
    grp_train.add_argument("--batch-size", type=int, default=None, help="Override batch size (Ignored)")
    grp_train.add_argument("--total-steps", type=int, default=None, help="Override total training steps (Ignored)")
    grp_train.add_argument("--save-every-n-steps", type=int, default=None, help="Checkpoint interval (Ignored)")
    grp_train.add_argument("--log-every-n-steps", type=int, default=None, help="Logging interval (Ignored)")
    grp_train.add_argument("--generate-every-n_steps", type=int, default=None, help="Generation sample interval (Ignored)")
    grp_train.add_argument("--wandb-project", type=str, default=None, help="WandB project name (Ignored)")
    grp_train.add_argument("--prefetch-batches", type=int, default=None, help="Number of batches to prefetch (Ignored)")
    grp_train.add_argument("--lr", type=float, default=None, dest="base_lr", help="Override learning rate (Ignored)")
    grp_train.add_argument("--no-remat", action="store_true", help="Disable gradient checkpointing (rematerialization)")

    return parser.parse_args()

args = parse_args()

import jax
import jax.numpy as jnp
# jax.distributed.initialize() # Not strictly necessary for parameter counting

from seqcond.config import Config, ModelConfig, TrainingConfig
from seqcond.jax.train import create_model_from_config
from seqcond.jax.model import init_model, count_parameters

def get_config(args) -> Config:
    """Builds the configuration object from arguments."""
    
    # 1. Base Model Config
    model_factory = getattr(ModelConfig, args.size)
    
    # Collect overrides for ModelConfig
    model_overrides = {}
    
    # Direct mappings (name matches)
    for field in ["num_layers", "d_model", "d_ff", "num_thetas", "seqcond_heads", "num_query_heads", "model_type", "seqcond_ratio", "expand_factor", "chunk_size"]:
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

    # 2. Training Config (Minimal needed for Config object structure)
    training_defaults = dict(
        batch_size=1,
        maxlen=args.maxlen,
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
        use_wandb=False, # Force False
        wandb_project="slm-training-len768",
    )

    training_config = TrainingConfig(**training_defaults)

    return Config(model=model_config, training=training_config)

def main():
    print("Initializing JAX for parameter counting...")
    
    config = get_config(args)
    
    print(f"Model Configuration: {config.model}")

    # Create model
    model = create_model_from_config(config.model)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(args.seed)
    # Using a small batch size for initialization to save memory
    input_shape = (1, config.model.maxlen) 
    
    print("Initializing model parameters...")
    variables = init_model(model, rng, input_shape=input_shape)
    params = variables["params"]
    
    # Count parameters
    total_params = count_parameters(params)
    
    print("-" * 40)
    print(f"Total Parameters: {total_params:,}")
    print("-" * 40)

if __name__ == "__main__":
    main()
