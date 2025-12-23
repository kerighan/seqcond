"""
Configuration dataclasses for model and training settings.
Shared between TensorFlow and JAX implementations.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Model type
    model_type: Literal["transformer", "seqcond"] = "seqcond"

    # Core architecture
    d_model: int = 768
    d_ff: int = 2304  # 3 * d_model
    num_layers: int = 12
    vocab_size: int = 100300
    maxlen: int = 768
    dropout: float = 0.0
    tie_weights: bool = True

    # Transformer params
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # For GQA, None = MHA
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6

    # SeqCond params
    seqcond_heads: int = 32
    num_thetas: int = 4
    derivative_order: int = 0
    num_anchor_heads: int = 0
    use_conv: bool = True
    conv_kernel_size: int = 4
    use_positional_embedding: bool = False
    seqcond_ratio: int = 5  # Interleaving ratio

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @property
    def name(self) -> str:
        """Generate a descriptive name for the model."""
        if self.model_type == "transformer":
            base = f"{self.model_type}-l{self.num_layers}-d{self.d_model}-h{self.num_heads}"
        elif self.model_type == "seqcond":
            base = f"{self.model_type}-l{self.num_layers}-d{self.d_model}"
            base += f"-th{self.num_heads}-sh{self.seqcond_heads}"
            base += f"-m{self.num_thetas}-r{self.seqcond_ratio}"
            base += f"-o{self.derivative_order}-a{self.num_anchor_heads}"
        else:
            base = f"{self.model_type}-l{self.num_layers}-d{self.d_model}"
        return base

    @classmethod
    def small(cls, **kwargs) -> "ModelConfig":
        """Small model (~30M params). Override any param with kwargs."""
        defaults = dict(
            d_model=768,
            d_ff=768 * 3,
            num_layers=12,
            num_heads=12,
            seqcond_heads=32,
            num_kv_heads=4,
            num_thetas=2,
            seqcond_ratio=5,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def medium(cls, **kwargs) -> "ModelConfig":
        """Medium model (~110M params). Override any param with kwargs."""
        defaults = dict(
            d_model=960,
            d_ff=960 * 3,
            num_layers=32,
            num_heads=15,
            seqcond_heads=30,
            num_kv_heads=5,
            num_thetas=2,
            seqcond_ratio=7,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def large(cls, **kwargs) -> "ModelConfig":
        """Large model (~350M params). Override any param with kwargs."""
        defaults = dict(
            d_model=1024,
            d_ff=1024 * 4,
            num_layers=24,
            num_heads=16,
            seqcond_heads=64,
            num_kv_heads=4,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def xlarge(cls, **kwargs) -> "ModelConfig":
        """XLarge model (~770M params). Override any param with kwargs."""
        defaults = dict(
            d_model=1536,
            d_ff=1536 * 4,
            num_layers=24,
            num_heads=16,
            seqcond_heads=96,
            num_kv_heads=8,
        )
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Batch and sequence
    batch_size: int = 1
    maxlen: int = 768

    # Optimizer
    base_lr: float = 1e-3
    warmup_steps: int = 100
    total_steps: int = 100000
    weight_decay: float = 1e-2
    clipnorm: float = 1.0
    beta_1: float = 0.9
    beta_2: float = 0.999

    # Gradient accumulation
    grad_accum_steps: int = 1
    use_multiple_tpus: bool = False
    train_thetas: bool = True

    # Mixed precision
    mixed_precision: Optional[Literal["float16", "bfloat16"]] = None

    # Mixed precision behavior
    keep_weights_fp32: bool = False

    # Logging
    log_every_n_steps: int = 100
    generate_every_n_steps: int = 1000
    save_every_n_steps: int = 5000

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "slm-training"

    # Paths
    checkpoint_dir: str = "checkpoints"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def fast(cls, **kwargs) -> "TrainingConfig":
        """Fast training config for testing. Override any param with kwargs."""
        defaults = dict(
            batch_size=1,
            total_steps=1000,
            log_every_n_steps=50,
            generate_every_n_steps=200,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def default(cls, **kwargs) -> "TrainingConfig":
        """Default training config. Override any param with kwargs."""
        return cls(**kwargs)

    @classmethod
    def long(cls, **kwargs) -> "TrainingConfig":
        """Long training config for full runs. Override any param with kwargs."""
        defaults = dict(
            total_steps=500000,
            warmup_steps=1000,
            save_every_n_steps=10000,
        )
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class Config:
    """Combined model and training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        """Convert full config to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    @property
    def name(self) -> str:
        """Delegate to model.name."""
        return self.model.name

    @classmethod
    def small(cls, **model_kwargs) -> "Config":
        """Small model with default training. Override model params with kwargs."""
        return cls(model=ModelConfig.small(**model_kwargs))

    @classmethod
    def medium(cls, **model_kwargs) -> "Config":
        """Medium model with default training. Override model params with kwargs."""
        return cls(model=ModelConfig.medium(**model_kwargs))

    @classmethod
    def large(cls, **model_kwargs) -> "Config":
        """Large model with default training. Override model params with kwargs."""
        return cls(model=ModelConfig.large(**model_kwargs))

    @classmethod
    def test(cls, **model_kwargs) -> "Config":
        """Small model with fast training for testing. Override model params with kwargs."""
        return cls(
            model=ModelConfig.small(**model_kwargs),
            training=TrainingConfig.fast(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Reconstruct Config from a dictionary (e.g., from checkpoint)."""
        model_dict = d.get("model", {})
        training_dict = d.get("training", {})
        return cls(
            model=ModelConfig(**model_dict),
            training=TrainingConfig(**training_dict),
        )
