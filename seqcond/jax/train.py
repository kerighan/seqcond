"""
High-level training utilities for JAX/Flax models.
"""

import os
import time
import pickle
import itertools
from typing import Any, Optional, Callable, Tuple, Dict

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.core import FrozenDict
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit


from .model import (
    create_seqcond_model,
    create_seqcond_model_v2,
    create_mamba_model,
    create_transformer_model,
    create_bivector_model,
    create_rwkv_model,
    create_rwkv_hybrid_model,
    create_optimizer,
    warmup_cosine_decay_schedule,
    init_model,
    count_parameters,
    sparse_categorical_crossentropy_loss,
)
from .metrics import MetricsAccumulator, compute_batch_metrics
from .callback import generate_text

from ..config import ModelConfig, TrainingConfig, Config
from ..dataset import DataLoader, create_tf_dataset


def _tree_map_with_names(tree, fn: Callable, path=()):
    if isinstance(tree, FrozenDict):
        return FrozenDict(
            {k: _tree_map_with_names(v, fn, path + (k,)) for k, v in tree.items()}
        )
    if isinstance(tree, dict):
        return {k: _tree_map_with_names(v, fn, path + (k,)) for k, v in tree.items()}
    return fn(path, tree)


def _theta_trainable_mask(params, train_thetas: bool):
    if train_thetas:
        return jax.tree_util.tree_map(lambda _: True, params)

    def selector(path, _):
        return not any("theta" in str(key).lower() for key in path)

    return _tree_map_with_names(params, selector)


def _create_grad_mask(params, train_thetas: bool):
    bool_mask = _theta_trainable_mask(params, train_thetas)

    def to_array(keep, p):
        ones = jnp.ones_like(p)
        return ones if keep else jnp.zeros_like(p)

    return jax.tree_util.tree_map(to_array, bool_mask, params)


def _apply_grad_mask(grads, mask):
    if mask is None:
        return grads
    return jax.tree_util.tree_map(lambda g, m: g * m, grads, mask)


def _global_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    leaves = [jnp.asarray(x, dtype=jnp.float32) for x in leaves]
    return jnp.sqrt(sum([jnp.sum(x * x) for x in leaves]))


def create_model_from_config(config: ModelConfig):
    """Create a model from a ModelConfig."""
    if config.model_type == "transformer":
        return create_transformer_model(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            vocab_size=config.vocab_size,
            maxlen=config.maxlen,
            dropout=config.dropout,
            tie_weights=config.tie_weights,
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
            remat=config.remat,
        )
    elif config.model_type == "bivector":
        return create_bivector_model(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            vocab_size=config.vocab_size,
            maxlen=config.maxlen,
            dropout=config.dropout,
            tie_weights=config.tie_weights,
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
            remat=config.remat,
        )
    elif config.model_type == "seqcond":
        return create_seqcond_model(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            maxlen=config.maxlen,
            use_positional_embedding=config.use_positional_embedding,
            seqcond_ratio=config.seqcond_ratio,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            seqcond_heads=config.seqcond_heads,
            num_query_heads=config.num_query_heads,
            num_anchor_heads=config.num_anchor_heads,
            num_thetas=config.num_thetas,
            derivative_order=config.derivative_order,
            expand_factor=config.expand_factor,
            dropout=config.dropout,
            tie_weights=config.tie_weights,
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
            conv_kernel_size=config.conv_kernel_size,
            remat=config.remat,
            chunk_size=config.chunk_size,
            use_square_matrix=config.use_square_matrix,
        )
    elif config.model_type == "seqcond2":
        return create_seqcond_model_v2(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            maxlen=config.maxlen,
            use_positional_embedding=config.use_positional_embedding,
            seqcond_ratio=config.seqcond_ratio,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            seqcond_heads=config.seqcond_heads,
            num_anchor_heads=config.num_anchor_heads,
            num_thetas=config.num_thetas,
            expand_factor=config.expand_factor,
            dropout=config.dropout,
            tie_weights=config.tie_weights,
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
            conv_kernel_size=config.conv_kernel_size,
            remat=config.remat,
        )
    elif config.model_type == "mamba":
        return create_mamba_model(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            maxlen=config.maxlen,
            seqcond_ratio=config.seqcond_ratio,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            state_size=config.state_size,
            expand_factor=config.expand_factor,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
            tie_weights=config.tie_weights,
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
            remat=config.remat,
        )
    elif config.model_type == "rwkv":
        # Check if we should create a hybrid model (ratio > 0 means interleave Transformer layers)
        if config.seqcond_ratio > 0:
            return create_rwkv_hybrid_model(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                rwkv_ratio=config.seqcond_ratio,
                maxlen=config.maxlen,
                dropout=config.dropout,
                tie_weights=config.tie_weights,
                qk_norm=config.qk_norm,
                qk_norm_eps=config.qk_norm_eps,
                remat=config.remat,
                use_scan=True,
                dtype=jnp.bfloat16,
            )
        else:
            # Pure RWKV model
            return create_rwkv_model(
                vocab_size=config.vocab_size,
                n_embd=config.d_model,
                n_layer=config.num_layers,
                use_scan=True,
                remat=config.remat,
                dtype=jnp.bfloat16,
            )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def get_compute_dtype(mixed_precision: Optional[str]) -> jnp.dtype:
    """Get compute dtype from mixed precision setting."""
    if mixed_precision == "bfloat16":
        return jnp.bfloat16
    elif mixed_precision == "float16":
        return jnp.float16
    return jnp.float32


def cast_params_to_dtype(params: Any, dtype: jnp.dtype) -> Any:
    """Cast all parameters to a specific dtype."""
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), params)


def make_train_step(model, optimizer, compute_dtype=jnp.float32, grad_mask=None):
    """Create a JIT-compiled train step function."""

    compute_dtype = jnp.dtype(compute_dtype)
    keep_weights_fp32 = compute_dtype != jnp.float32

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            p_apply = cast_params_to_dtype(p, compute_dtype) if keep_weights_fp32 else p
            logits = model.apply({"params": p_apply}, x, deterministic=True)
            loss = sparse_categorical_crossentropy_loss(logits, y, ignore_class=0)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = _apply_grad_mask(grads, grad_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics = compute_batch_metrics(logits, y, ignore_class=0)
        return new_params, new_opt_state, metrics

    return train_step


def make_train_step_with_debug(
    model, optimizer, lr_schedule, clipnorm: float, compute_dtype=jnp.float32
):
    """Train step that also returns debug scalars (grad/updates norms, LR, clip ratio)."""

    clipnorm = float(clipnorm)
    compute_dtype = jnp.dtype(compute_dtype)
    keep_weights_fp32 = compute_dtype != jnp.float32

    @jax.jit
    def train_step(params, opt_state, x, y, step):
        def loss_fn(p):
            p_apply = cast_params_to_dtype(p, compute_dtype) if keep_weights_fp32 else p
            logits = model.apply({"params": p_apply}, x, deterministic=True)
            loss = sparse_categorical_crossentropy_loss(logits, y, ignore_class=0)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        grad_norm = _global_norm(grads)
        clip_ratio = jnp.minimum(1.0, clipnorm / (grad_norm + 1e-6))

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        update_norm = _global_norm(updates)

        new_params = optax.apply_updates(params, updates)
        lr = lr_schedule(step)

        debug = {
            "lr": lr,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "clip_ratio": clip_ratio,
        }
        return new_params, new_opt_state, loss, logits, debug

    return train_step


def make_fsdp_train_step(model, optimizer, compute_dtype=jnp.float32, grad_mask=None):
    """Create a PJIT-compiled FSDP train step."""

    compute_dtype = jnp.dtype(compute_dtype)
    keep_weights_fp32 = compute_dtype != jnp.float32

    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            p_apply = cast_params_to_dtype(p, compute_dtype) if keep_weights_fp32 else p
            logits = model.apply({"params": p_apply}, x, deterministic=True)
            loss = sparse_categorical_crossentropy_loss(logits, y, ignore_class=0)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = _apply_grad_mask(grads, grad_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics = compute_batch_metrics(logits, y, ignore_class=0)
        return new_params, new_opt_state, metrics

    return train_step


def make_pmap_train_step(
    model, optimizer, compute_dtype=jnp.float32, axis_name: str = "devices"
):
    """Create a pmapped data-parallel train step."""

    compute_dtype = jnp.dtype(compute_dtype)
    keep_weights_fp32 = compute_dtype != jnp.float32

    def train_step(params, opt_state, x, y):
        def loss_fn(p, xb, yb):
            p_apply = cast_params_to_dtype(p, compute_dtype) if keep_weights_fp32 else p
            logits = model.apply({"params": p_apply}, xb, deterministic=True)
            loss = sparse_categorical_crossentropy_loss(logits, yb, ignore_class=0)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
        loss = jax.lax.pmean(loss, axis_name=axis_name)
        grads = jax.lax.pmean(grads, axis_name=axis_name)

        # Compute metrics locally
        metrics_local = compute_batch_metrics(logits, y, ignore_class=0)
        # Aggregate across devices
        metrics = {
            "correct": jax.lax.psum(metrics_local["correct"], axis_name=axis_name),
            "count": jax.lax.psum(metrics_local["count"], axis_name=axis_name),
            "loss": jax.lax.psum(metrics_local["loss"], axis_name=axis_name),
        }

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    return jax.pmap(train_step, axis_name=axis_name)


def make_grad_step(model):
    """Create a JIT-compiled gradient computation step (for accumulation)."""

    compute_dtype = getattr(model, "compute_dtype", None)
    compute_dtype = jnp.float32 if compute_dtype is None else jnp.dtype(compute_dtype)
    keep_weights_fp32 = compute_dtype != jnp.float32

    @jax.jit
    def grad_step(params, x, y):
        def loss_fn(p):
            p_apply = cast_params_to_dtype(p, compute_dtype) if keep_weights_fp32 else p
            logits = model.apply({"params": p_apply}, x, deterministic=True)
            loss = sparse_categorical_crossentropy_loss(logits, y, ignore_class=0)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        metrics = compute_batch_metrics(logits, y, ignore_class=0)
        return metrics, grads

    return grad_step


def make_update_step(optimizer):
    """Create a JIT-compiled parameter update step."""

    @jax.jit
    def update_step(params, opt_state, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    return update_step


def accumulate_grads(grads_accum, grads, accum_steps: int):
    """Accumulate gradients."""
    if grads_accum is None:
        return jax.tree_util.tree_map(lambda g: g / accum_steps, grads)
    return jax.tree_util.tree_map(
        lambda acc, g: acc + g / accum_steps, grads_accum, grads
    )


def save_checkpoint(
    params: Any,
    opt_state: Any,
    config: Config,
    path: str,
    step: Optional[int] = None,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    params_cpu = jax.device_get(params)
    opt_state_cpu = jax.device_get(opt_state) if opt_state is not None else None
    data = {
        "params": params_cpu,
        "opt_state": opt_state_cpu,
        "config": config.to_dict(),
        "step": step,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(path: str) -> Tuple[Any, Dict, Optional[int], Optional[Any]]:
    """Load model checkpoint. Returns (params, config_dict, step, opt_state)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["params"], data["config"], data.get("step"), data.get("opt_state")


class Trainer:
    """High-level trainer for JAX models."""

    def __init__(
        self,
        config: Config,
        data_loader: DataLoader = None,
        tokenizer: Any = None,
        model_name: Optional[str] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        self.config = config
        self.model_config = config.model
        self.train_config = config.training
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.model_name = model_name or config.name
        self.resume_checkpoint = resume_checkpoint
        self._wandb = None

        # Will be initialized in setup()
        self.model = None
        self.params = None
        self._base_optimizer = None
        self.optimizer = None
        self.opt_state = None
        self.compute_dtype = None
        self.start_step = 0
        self.use_pmap = False
        self.use_fsdp = False
        self.num_devices = jax.device_count()
        self.per_device_batch = None
        self._pmap_train_step = None
        self._fsdp_train_step = None
        self.mesh = None
        self.params_sharding = None
        self.data_sharding = None

    def _replicate_tree(self, tree):
        return jax.device_put_replicated(tree, jax.local_devices())

    def _unreplicate_tree(self, tree):
        return jax.tree_util.tree_map(lambda x: x[0], tree)

    def _unshard_tree(self, tree):
        """Unshard a tree from devices to host."""
        return jax.tree_util.tree_map(
            lambda x: x.addressable_data(0),
            tree,
        )

    def _host_tree(self, tree):
        if self.use_fsdp:
            return self._unshard_tree(tree)
        return jax.device_get(tree)

    def _params_for_host(self):
        params = self._host_tree(self.params)
        if self.use_pmap:
            params = self._unreplicate_tree(params)
        return params

    def _opt_state_for_host(self):
        opt_state = (
            self._host_tree(self.opt_state) if self.opt_state is not None else None
        )
        if opt_state is not None and self.use_pmap:
            opt_state = self._unreplicate_tree(opt_state)
        return opt_state

    def setup(self, seed: int = 42):
        """Initialize model, optimizer, and compile training steps."""
        print("=" * 60)
        print("JAX Trainer Setup")
        print("=" * 60)

        # Set compute dtype
        self.compute_dtype = get_compute_dtype(self.train_config.mixed_precision)
        if self.compute_dtype != jnp.float32:
            print(f"\nUsing mixed precision: {self.train_config.mixed_precision}")
        else:
            print("\nUsing full precision: float32")

        # Create model
        print("Creating model...")
        self.model = create_model_from_config(self.model_config)

        # Create optimizer
        self._base_optimizer = create_optimizer(
            base_lr=self.train_config.base_lr,
            warmup_steps=self.train_config.warmup_steps,
            total_steps=self.train_config.total_steps,
            weight_decay=self.train_config.weight_decay,
            clipnorm=self.train_config.clipnorm,
            beta_1=self.train_config.beta_1,
            beta_2=self.train_config.beta_2,
        )
        self.optimizer = self._base_optimizer

        # Determine multi-device usage
        self.use_fsdp = (
            bool(self.train_config.full_shard_data_parallel) and self.num_devices > 1
        )
        self.use_pmap = (
            bool(self.train_config.use_multiple_tpus)
            and not self.use_fsdp
            and self.num_devices > 1
        )

        if self.train_config.use_multiple_tpus and not (self.use_pmap or self.use_fsdp):
            print(
                "Warning: use_multiple_tpus is enabled but only one device was detected. "
                "Falling back to single-device training."
            )

        grad_accum_steps = self.train_config.grad_accum_steps
        if self.use_pmap:
            if grad_accum_steps != 1:
                raise ValueError(
                    "Gradient accumulation is not supported with pmap multi-device training. Use FSDP instead."
                )
            if self.train_config.batch_size % self.num_devices != 0:
                raise ValueError(
                    "Batch size must be divisible by the number of devices for multi-TPU training."
                )
            self.per_device_batch = self.train_config.batch_size // self.num_devices

        if self.use_fsdp:
            if self.train_config.batch_size % self.num_devices != 0:
                raise ValueError(
                    "Batch size must be divisible by the number of devices for FSDP training."
                )
            self.per_device_batch = self.train_config.batch_size // self.num_devices
            print(
                f"Using FSDP across {self.num_devices} devices "
                f"(per-device batch size = {self.per_device_batch})."
            )
            self.mesh = Mesh(jax.devices(), axis_names=("dp",))

            # Determine sharding strategy based on parameter shapes
            # We use eval_shape to get the structure and shapes without allocating memory
            rng_init = jax.random.PRNGKey(0)
            abstract_variables = jax.eval_shape(
                lambda r: self.model.init(r, jnp.ones((1, 1), dtype=jnp.int32)),
                rng_init,
            )
            abstract_params = abstract_variables["params"]

            def get_sharding(x):
                # Shard on first dimension if it's divisible by num_devices
                if x.ndim > 0 and x.shape[0] % self.num_devices == 0:
                    return NamedSharding(self.mesh, PartitionSpec("dp"))
                # Otherwise replicate
                return NamedSharding(self.mesh, PartitionSpec())

            self.params_sharding = jax.tree_util.tree_map(get_sharding, abstract_params)

            # Compute sharding for optimizer state
            abstract_opt_state = jax.eval_shape(self.optimizer.init, abstract_params)
            self.opt_state_sharding = jax.tree_util.tree_map(
                get_sharding, abstract_opt_state
            )

            # Data is sharded on batch dimension
            self.data_sharding = NamedSharding(self.mesh, PartitionSpec("dp"))
            # Scalar metrics must be replicated
            replicated_sharding = NamedSharding(self.mesh, PartitionSpec())

            metrics_sharding = {
                "correct": replicated_sharding,
                "count": replicated_sharding,
                "loss": replicated_sharding,
            }

            # Init params and opt_state with PJIT
            def init_fn(rng, model, optimizer, input_shape):
                variables = model.init(rng, jnp.ones(input_shape, dtype=jnp.int32))
                params = variables["params"]
                opt_state = optimizer.init(params)
                return params, opt_state

            pjit_init_fn = pjit(
                init_fn,
                static_argnums=(1, 2, 3),
                in_shardings=(None,),
                out_shardings=(self.params_sharding, self.opt_state_sharding),
            )

            rng = jax.random.PRNGKey(seed)
            with self.mesh:
                self.params, self.opt_state = pjit_init_fn(
                    rng,
                    self.model,
                    self._base_optimizer,
                    (self.train_config.batch_size, self.model_config.maxlen),
                )

        else:  # Single device or PMAP
            rng = jax.random.PRNGKey(seed)
            variables = init_model(
                self.model, rng, input_shape=(1, self.model_config.maxlen)
            )
            self.params = variables["params"]

        # Cast to compute dtype if not keeping weights in FP32
        if (
            self.compute_dtype != jnp.float32
            and not self.train_config.keep_weights_fp32
            and not self.use_fsdp  # FSDP handles this internally
        ):
            self.params = cast_params_to_dtype(self.params, self.compute_dtype)

        try:
            setattr(self.model, "compute_dtype", self.compute_dtype)
        except Exception:
            pass

        # Use PJIT to count params if FSDP, otherwise standard
        if self.use_fsdp:

            def count_params_fn(p):
                return sum(x.size for x in jax.tree_util.tree_leaves(p))

            num_params = pjit(
                count_params_fn,
                in_shardings=(self.params_sharding,),
                out_shardings=None,
            )(self.params)
        else:
            num_params = count_parameters(self.params)

        print(f"Model type: {self.model_config.model_type}")
        print(f"Parameters: {num_params:,}")

        grad_mask = _create_grad_mask(self.params, self.train_config.train_thetas)
        self._grad_mask = grad_mask or None

        if not self.use_fsdp:
            self.opt_state = self.optimizer.init(self.params)

        if self.use_pmap:
            print(
                f"Using data parallelism (pmap) across {self.num_devices} devices "
                f"(per-device batch size = {self.per_device_batch})."
            )

        # Resume from checkpoint if provided
        if self.resume_checkpoint:
            if os.path.exists(self.resume_checkpoint):
                (
                    ckpt_params,
                    _,
                    ckpt_step,
                    ckpt_opt_state,
                ) = load_checkpoint(self.resume_checkpoint)
                self.params = ckpt_params
                if ckpt_opt_state is not None:
                    self.opt_state = ckpt_opt_state
                self.start_step = ckpt_step or 0
                print(
                    f"Resumed from checkpoint {self.resume_checkpoint} "
                    f"(step {self.start_step})"
                )

                # Validate parameter shapes
                try:
                    emb_shape = self.params["token_embedding"]["embedding"].shape
                    expected_shape = (
                        self.model_config.vocab_size,
                        self.model_config.d_model,
                    )
                    if emb_shape != expected_shape:
                        print(f"WARNING: Parameter shape mismatch in token_embedding!")
                        print(f"  Checkpoint: {emb_shape}")
                        print(f"  Config:     {expected_shape}")
                        print(
                            f"  This may cause errors. Consider updating config.vocab_size to {emb_shape[0]}."
                        )
                except KeyError:
                    pass
            else:
                print(
                    f"Warning: checkpoint {self.resume_checkpoint} not found. "
                    "Starting from scratch."
                )

        if self.use_pmap:
            self.params = self._replicate_tree(self.params)
            self.opt_state = self._replicate_tree(self.opt_state)

        # Create JIT-compiled steps
        if self.use_fsdp:
            if grad_accum_steps > 1:
                # 1. Grad step for FSDP
                def fsdp_grad_step_fn(params, x, y):
                    def loss_fn(p):
                        p_apply = (
                            cast_params_to_dtype(p, self.compute_dtype)
                            if (
                                self.compute_dtype != jnp.float32
                                and self.train_config.keep_weights_fp32
                            )
                            else p
                        )
                        logits = self.model.apply(
                            {"params": p_apply}, x, deterministic=True
                        )
                        loss = sparse_categorical_crossentropy_loss(
                            logits, y, ignore_class=0
                        )
                        return loss, logits

                    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        params
                    )
                    grads = _apply_grad_mask(grads, self._grad_mask)
                    metrics = compute_batch_metrics(logits, y, ignore_class=0)
                    return grads, metrics

                self._fsdp_grad_step = pjit(
                    fsdp_grad_step_fn,
                    in_shardings=(
                        self.params_sharding,
                        self.data_sharding,
                        self.data_sharding,
                    ),
                    out_shardings=(self.params_sharding, metrics_sharding),
                )

                # 2. Update step for FSDP
                def fsdp_update_step_fn(params, opt_state, grads):
                    updates, new_opt_state = self.optimizer.update(
                        grads, opt_state, params
                    )
                    new_params = optax.apply_updates(params, updates)
                    return new_params, new_opt_state

                self._fsdp_update_step = pjit(
                    fsdp_update_step_fn,
                    in_shardings=(
                        self.params_sharding,
                        self.opt_state_sharding,
                        self.params_sharding,
                    ),
                    out_shardings=(self.params_sharding, self.opt_state_sharding),
                )
                print(
                    f"Using FSDP with gradient accumulation: {grad_accum_steps} steps"
                )
            else:
                step_fn = make_fsdp_train_step(
                    self.model, self.optimizer, self.compute_dtype, self._grad_mask
                )
                self._fsdp_train_step = pjit(
                    step_fn,
                    in_shardings=(
                        self.params_sharding,
                        self.opt_state_sharding,
                        self.data_sharding,
                        self.data_sharding,
                    ),
                    out_shardings=(
                        self.params_sharding,
                        self.opt_state_sharding,
                        metrics_sharding,
                    ),
                )
        elif self.use_pmap:
            self._pmap_train_step = make_pmap_train_step(
                self.model, self.optimizer, self.compute_dtype
            )
        elif grad_accum_steps > 1:
            self._grad_step = make_grad_step(self.model)
            self._update_step = make_update_step(self.optimizer)
            print(f"Using gradient accumulation: {grad_accum_steps} steps")
        else:
            debug_opt = os.environ.get("SLM_JAX_DEBUG_OPT", "0").strip() == "1"
            if debug_opt:
                lr_schedule = warmup_cosine_decay_schedule(
                    base_lr=self.train_config.base_lr,
                    warmup_steps=self.train_config.warmup_steps,
                    total_steps=self.train_config.total_steps,
                )
                self._train_step_dbg = make_train_step_with_debug(
                    self.model,
                    self.optimizer,
                    lr_schedule,
                    self.train_config.clipnorm,
                    self.compute_dtype,
                )
                print("JAX debug optimizer logging enabled (SLM_JAX_DEBUG_OPT=1)")
            else:
                self._train_step = make_train_step(
                    self.model, self.optimizer, self.compute_dtype, self._grad_mask
                )

        # Initialize wandb if enabled
        if self.train_config.use_wandb:
            self._init_wandb(num_params)

        return self

    def _init_wandb(self, num_params: int):
        """Initialize wandb logging."""
        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.train_config.wandb_project,
                name=self.model_name,
                config={
                    **self.config.to_dict(),
                    "n_trainable_params": num_params,
                },
            )
            print(
                f"Wandb initialized: {self.train_config.wandb_project}/{self.model_name}"
            )
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")
            self._wandb = None

    def train(self):
        """Run training loop."""
        tc = self.train_config
        grad_accum_steps = tc.grad_accum_steps

        # Create data loader if not provided
        # Use tf.data for better performance on TPUs
        if self.data_loader is None:
            micro_steps = tc.total_steps * grad_accum_steps
            print("Initializing tf.data pipeline...")
            dataset = create_tf_dataset(
                batch_size=tc.batch_size,
                max_steps=micro_steps,
                maxlen=tc.maxlen,
                prefetch_buffer=(
                    tc.prefetch_batches if tc.prefetch_batches > 0 else None
                ),
            )
            data_iterator = dataset.as_numpy_iterator()
            # We need to track tokens manually since we are not using the python DataLoader class
            tokens_seen = 0
            last_tokens_seen = 0
            using_tf_data = True
        else:
            data_iterator = iter(self.data_loader)
            using_tf_data = False

        # Training state
        metrics = MetricsAccumulator(ignore_class=0)
        grads_accum = None
        metrics_accum = None
        accum_count = 0
        macro_step = self.start_step

        if macro_step > 0:
            print(f"Continuing training from step {macro_step}")

        start_time = time.time()
        last_log_time = start_time
        if not using_tf_data:
            self.data_loader.reset_stats()

        print("\nStarting training...")

        for step in itertools.count(start=1):
            if step > tc.total_steps * grad_accum_steps:
                break

            try:
                x_batch, y_batch = next(data_iterator)
                # Update manual token tracking if using tf.data
                if using_tf_data:
                    tokens_in_batch = x_batch.size
                    tokens_seen += tokens_in_batch
            except StopIteration:
                print("Data loader exhausted.")
                break

            metrics_batch = None

            if self.use_fsdp:
                x = jax.device_put(
                    jnp.array(x_batch, dtype=jnp.int32), self.data_sharding
                )
                y = jax.device_put(
                    jnp.array(y_batch, dtype=jnp.int32), self.data_sharding
                )

                if grad_accum_steps > 1:
                    with self.mesh:
                        grads, metrics_step = self._fsdp_grad_step(self.params, x, y)

                    grads_accum = accumulate_grads(grads_accum, grads, grad_accum_steps)

                    if metrics_accum is None:
                        metrics_accum = metrics_step
                    else:
                        metrics_accum = jax.tree_util.tree_map(
                            jnp.add, metrics_accum, metrics_step
                        )

                    accum_count += 1

                    if accum_count >= grad_accum_steps:
                        with self.mesh:
                            self.params, self.opt_state = self._fsdp_update_step(
                                self.params, self.opt_state, grads_accum
                            )
                        grads_accum = None
                        metrics_batch = metrics_accum
                        metrics_accum = None
                        accum_count = 0
                        macro_step += 1
                else:
                    with self.mesh:
                        (
                            self.params,
                            self.opt_state,
                            metrics_batch,
                        ) = self._fsdp_train_step(self.params, self.opt_state, x, y)
                    macro_step += 1

            elif self.use_pmap:
                x = jnp.array(x_batch, dtype=jnp.int32).reshape(
                    self.num_devices, self.per_device_batch, -1
                )
                y = jnp.array(y_batch, dtype=jnp.int32).reshape(
                    self.num_devices, self.per_device_batch, -1
                )
                self.params, self.opt_state, metrics_batch = self._pmap_train_step(
                    self.params, self.opt_state, x, y
                )
                # PMAP returns sharded arrays (one per device). Since we psum-ed, they are identical.
                # Take the first device's output.
                metrics_batch = jax.tree_util.tree_map(lambda x: x[0], metrics_batch)
                macro_step += 1

            else:
                x = jnp.array(x_batch, dtype=jnp.int32)
                y = jnp.array(y_batch, dtype=jnp.int32)

                if grad_accum_steps > 1:
                    metrics_step, grads = self._grad_step(self.params, x, y)
                    grads_accum = accumulate_grads(grads_accum, grads, grad_accum_steps)

                    if metrics_accum is None:
                        metrics_accum = metrics_step
                    else:
                        metrics_accum = jax.tree_util.tree_map(
                            jnp.add, metrics_accum, metrics_step
                        )

                    accum_count += 1
                    if accum_count >= grad_accum_steps:
                        self.params, self.opt_state = self._update_step(
                            self.params, self.opt_state, grads_accum
                        )
                        grads_accum = None
                        metrics_batch = metrics_accum
                        metrics_accum = None
                        accum_count = 0
                        macro_step += 1
                else:
                    if self._train_step_dbg:
                        # Fallback for debug mode (not updated to metrics dict yet)
                        # We just ignore metrics here or crash if used.
                        # Assuming user won't use debug_opt=1 for performance run.
                        self.params, self.opt_state, loss, logits, dbg = (
                            self._train_step_dbg(
                                self.params, self.opt_state, x, y, macro_step + 1
                            )
                        )
                        metrics_batch = compute_batch_metrics(logits, y, ignore_class=0)
                    else:
                        self.params, self.opt_state, metrics_batch = self._train_step(
                            self.params, self.opt_state, x, y
                        )
                    macro_step += 1

            # Update metrics accumulator
            if metrics_batch is not None:
                metrics.update_with_precomputed(metrics_batch)

                # Logging, Generation, Checkpointing
                if macro_step > 0 and macro_step % tc.log_every_n_steps == 0:
                    if using_tf_data:
                        tokens_delta = tokens_seen - last_tokens_seen
                        last_tokens_seen = tokens_seen
                        current_tokens_seen = tokens_seen
                    else:
                        tokens_delta = self.data_loader.tokens_since_last_check()
                        current_tokens_seen = self.data_loader.tokens_seen

                    last_log_time = self._log_progress(
                        macro_step,
                        tc.total_steps,
                        metrics,
                        start_time,
                        last_log_time,
                        tokens_delta,
                        current_tokens_seen,
                    )
                    metrics.reset()

                if macro_step > 0 and macro_step % tc.generate_every_n_steps == 0:
                    self._generate_sample(macro_step)

                if (
                    macro_step > 0
                    and tc.save_every_n_steps > 0
                    and macro_step % tc.save_every_n_steps == 0
                ):
                    self._save_checkpoint(macro_step)

        print("\nTraining complete!")
        self._save_checkpoint(macro_step, final=True)
        return self.params

    def _log_progress(
        self,
        macro_step: int,
        total_steps: int,
        metrics: MetricsAccumulator,
        start_time: float,
        last_log_time: float,
        tokens_delta: int = 0,
        tokens_seen: int = 0,
    ) -> float:
        """Log training progress."""
        # This is now the only place we should be calling device_get for metrics
        # We sync FIRST to ensure accurate timing (Sync-to-Sync)
        results = jax.device_get(metrics.result())

        current_time = time.time()
        elapsed = current_time - last_log_time
        log_interval = self.train_config.log_every_n_steps
        batch_per_sec = log_interval / elapsed if elapsed > 0 else 0
        tokens_per_sec = tokens_delta / elapsed if elapsed > 0 else 0

        steps_remaining = total_steps - macro_step
        total_elapsed = current_time - start_time
        avg_time_per_step = total_elapsed / macro_step if macro_step > 0 else 1
        eta_seconds = steps_remaining * avg_time_per_step

        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        eta_s = int(eta_seconds % 60)

        print(
            f"Step {macro_step:6d}/{total_steps} | "
            f"loss: {results['loss']:.4f} | "
            f"acc: {results['acc']:.4f} | "
            f"ppx: {results['ppx']:.2f} | "
            f"{tokens_per_sec:,.0f} tok/s | "
            f"{tokens_seen:,} tokens | "
            f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        )

        # Wandb detailed logging
        if self._wandb is not None:
            log_data = {
                **{k: float(v) for k, v in results.items()},
                "tokens_per_sec": tokens_per_sec,
                "tokens_seen": tokens_seen,
            }
            self._wandb.log(log_data, step=macro_step)

        return current_time

    def _generate_sample(self, step: int):
        """Generate a text sample (dual pad modes: fixed, power2)."""
        # Use a complete prompt that expects assistant response with thinking
        prompt = "<|im_start|>user\n"

        # Ensure params are on the host for generation
        host_params = self._params_for_host()

        print(f"\n--- Generation at step {step} (fixed padding) ---")
        txt_fixed = generate_text(
            model=self.model,
            params=host_params,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            maxlen=self.model_config.maxlen,
            pad_mode="fixed",
            verbose=True,
        )

        print("\n-----\n")

        print(f"--- Generation at step {step} (power2 padding) ---")
        txt_power2 = generate_text(
            model=self.model,
            params=host_params,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            maxlen=self.model_config.maxlen,
            pad_mode="power2",
            verbose=True,
        )

        # Optional: could log txt_fixed/txt_power2 if needed
        _ = (txt_fixed, txt_power2)

    def _save_checkpoint(self, step: int, final: bool = False):
        """Save a checkpoint."""
        if final:
            path = f"{self.train_config.checkpoint_dir}/{self.model_name}.pkl"
        else:
            path = (
                f"{self.train_config.checkpoint_dir}/{self.model_name}_step{step}.pkl"
            )

        # Get params and opt_state from devices before saving
        params_to_save = self._params_for_host()
        opt_state_to_save = self._opt_state_for_host()

        save_checkpoint(params_to_save, opt_state_to_save, self.config, path, step)
        print(f"Checkpoint saved: {path}")


def train(
    config: Config,
    data_loader: DataLoader = None,
    tokenizer: Any = None,
    model_name: Optional[str] = None,
    seed: int = 42,
    resume_checkpoint: Optional[str] = None,
) -> Any:
    """
    High-level training function.

    Args:
        config: Combined model and training configuration
        data_loader: DataLoader instance (optional, created from config if None)
        tokenizer: Tokenizer with encode/decode methods (for generation)
        model_name: Name for checkpoints and logging
        seed: Random seed

    Returns:
        Trained model parameters
    """
    trainer = Trainer(
        config=config,
        data_loader=data_loader,
        tokenizer=tokenizer,
        model_name=model_name,
        resume_checkpoint=resume_checkpoint,
    )
    trainer.setup(seed=seed)
    return trainer.train()
