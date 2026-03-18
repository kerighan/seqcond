"""Quick test for the Depth-SeqCond model."""

import os
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import ops

from convert_torch_to_keras import (
    load_torch_checkpoint,
    build_keras_model,
    convert_weights,
)
from seqcond.keras3.depth_model import DepthSeqCondModel


def main():
    # 1. Load base model
    ckpt = "checkpoints/seqcond_lin5.pt"
    print(f"Loading base model from {ckpt}...")
    config, state_dict = load_torch_checkpoint(ckpt)
    base_model = build_keras_model(config)
    convert_weights(config, state_dict, base_model)

    # 2. Create depth model
    print("\nCreating DepthSeqCondModel...")
    model = DepthSeqCondModel(
        base_model,
        depth_num_heads=8,
        depth_num_query_heads=4,
        depth_num_thetas=4,
        depth_expand_factor=1.0,
        depth_out_expand_factor=3,
        depth_conv_kernel_size=4,
    )

    # 3. Forward pass
    print("\nForward pass with dummy input...")
    dummy = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    logits = model(dummy, training=False)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {logits.shape}")

    # 4. Count parameters
    total = model.count_params()
    base_total = base_model.count_params()
    depth_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    depth_non_trainable = total - depth_trainable

    print(f"\n  Base model params:      {base_total:>12,}")
    print(f"  Total params:           {total:>12,}")
    print(f"  Trainable (depth only): {depth_trainable:>12,}")
    print(f"  Frozen (base):          {depth_non_trainable:>12,}")
    print(f"  Depth overhead:         {100 * depth_trainable / base_total:.2f}%")

    # 5. List trainable weights
    print(f"\n  Trainable weights ({len(model.trainable_weights)}):")
    for w in model.trainable_weights:
        print(f"    {w.path:50s}  {str(w.shape):>20s}")

    # 6. Verify base is frozen
    base_trainable = [w for w in base_model.weights if w.trainable]
    print(f"\n  Base model trainable weights: {len(base_trainable)} (should be 0)")

    print("\n✓ Depth-SeqCond model works!")


if __name__ == "__main__":
    main()
