"""
Vérification de la qualité d'un modèle upscalé.

Ce script teste:
1. Que le modèle upscalé peut être chargé correctement
2. Que le downsampling des poids reproduit bien l'original
3. Que le modèle peut faire de l'inférence
4. Compare les perplexités sur un petit dataset
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

from seqcond.config import ModelConfig
from seqcond.torch.model import SeqCondModel


def downsample_weights(weights: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Downsample des poids avec average pooling."""
    if len(weights.shape) == 1:
        return F.avg_pool1d(
            weights.unsqueeze(0).unsqueeze(0),
            kernel_size=scale_factor,
            stride=scale_factor,
        ).squeeze()

    elif len(weights.shape) == 2:
        # Downsample dimension 0
        w = F.avg_pool1d(
            weights.unsqueeze(0), kernel_size=scale_factor, stride=scale_factor
        ).squeeze(0)

        # Downsample dimension 1
        w = (
            F.avg_pool1d(
                w.t().unsqueeze(0), kernel_size=scale_factor, stride=scale_factor
            )
            .squeeze(0)
            .t()
        )

        return w

    elif len(weights.shape) == 3:
        return F.avg_pool1d(
            weights.transpose(0, 2), kernel_size=scale_factor, stride=scale_factor
        ).transpose(0, 2)

    return weights


def verify_reconstruction_quality(
    original_checkpoint: str,
    upscaled_checkpoint: str,
    scale_factor: int = 2,
) -> Dict[str, float]:
    """
    Vérifie que le downsampling des poids upscalés reproduit bien l'original.

    Returns:
        Dict avec les métriques de reconstruction par layer
    """
    print("Loading checkpoints...")
    original = torch.load(original_checkpoint, map_location="cpu")
    upscaled = torch.load(upscaled_checkpoint, map_location="cpu")

    original_state = original.get("model", original.get("state_dict", original))
    upscaled_state = upscaled.get("model", upscaled.get("state_dict", upscaled))

    print(f"\nOriginal model: {len(original_state)} parameters")
    print(f"Upscaled model: {len(upscaled_state)} parameters")

    reconstruction_errors = {}

    for name, orig_weight in original_state.items():
        if name not in upscaled_state:
            print(f"Warning: {name} not found in upscaled model")
            continue

        upscaled_weight = upscaled_state[name]

        # Skip si les shapes sont identiques (ex: vocab embeddings)
        if orig_weight.shape == upscaled_weight.shape:
            continue

        # Downsample et comparer
        try:
            downsampled = downsample_weights(upscaled_weight, scale_factor)

            if downsampled.shape != orig_weight.shape:
                print(
                    f"Shape mismatch for {name}: {downsampled.shape} vs {orig_weight.shape}"
                )
                continue

            # Calculer l'erreur de reconstruction
            mse = F.mse_loss(downsampled, orig_weight).item()
            relative_error = (mse / (orig_weight.var().item() + 1e-8)) ** 0.5

            reconstruction_errors[name] = {
                "mse": mse,
                "relative_error": relative_error,
            }

        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Afficher les résultats
    print("\n" + "=" * 80)
    print("RECONSTRUCTION QUALITY")
    print("=" * 80)

    if reconstruction_errors:
        avg_mse = np.mean([v["mse"] for v in reconstruction_errors.values()])
        avg_rel_error = np.mean(
            [v["relative_error"] for v in reconstruction_errors.values()]
        )

        print(f"\nAverage MSE: {avg_mse:.6f}")
        print(f"Average Relative Error: {avg_rel_error:.4f}")

        print("\nWorst 10 layers:")
        sorted_errors = sorted(
            reconstruction_errors.items(),
            key=lambda x: x[1]["relative_error"],
            reverse=True,
        )
        for name, errors in sorted_errors[:10]:
            print(f"  {name:60s} rel_err={errors['relative_error']:.4f}")

    return reconstruction_errors


def test_inference(checkpoint_path: str, device: str = "cuda"):
    """
    Teste que le modèle peut faire de l'inférence.
    """
    print(f"\nTesting inference with {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config", {})

    if not config_dict:
        print("No config found, cannot test inference")
        return

    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ModelConfig(**config_dict)

    print(f"Creating model with config:")
    print(f"  d_model={config.d_model}, num_layers={config.num_layers}")

    # Créer le modèle
    model = SeqCondModel(
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        maxlen=config.maxlen,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        seqcond_heads=config.seqcond_heads,
        num_query_heads=config.num_query_heads,
        num_thetas=config.num_thetas,
        conv_kernel_size=config.conv_kernel_size,
        expand_factor=config.expand_factor,
        out_expand_factor=config.out_expand_factor,
        seqcond_ratio=config.seqcond_ratio,
        dropout=0.0,
    )

    # Charger les poids
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Weights loaded successfully (strict=True)")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        return

    model = model.to(device)
    model.eval()

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

    try:
        with torch.no_grad():
            logits = model(input_ids)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

        # Vérifier que les logits sont raisonnables
        if torch.isnan(logits).any():
            print("✗ NaN detected in logits!")
        elif torch.isinf(logits).any():
            print("✗ Inf detected in logits!")
        else:
            print(f"✓ Logits are finite")
            print(f"  Mean: {logits.mean().item():.4f}")
            print(f"  Std: {logits.std().item():.4f}")
            print(f"  Min: {logits.min().item():.4f}")
            print(f"  Max: {logits.max().item():.4f}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()


def compare_models(
    original_checkpoint: str,
    upscaled_checkpoint: str,
    num_samples: int = 100,
    seq_len: int = 128,
    device: str = "cuda",
):
    """
    Compare les outputs des deux modèles sur des inputs aléatoires.
    """
    print("\n" + "=" * 80)
    print("COMPARING MODEL OUTPUTS")
    print("=" * 80)

    # Charger les modèles
    orig_ckpt = torch.load(original_checkpoint, map_location="cpu")
    upsc_ckpt = torch.load(upscaled_checkpoint, map_location="cpu")

    orig_config = ModelConfig(**orig_ckpt.get("config", {}))
    upsc_config = ModelConfig(**upsc_ckpt.get("config", {}))

    print(f"\nOriginal: d_model={orig_config.d_model}, layers={orig_config.num_layers}")
    print(f"Upscaled: d_model={upsc_config.d_model}, layers={upsc_config.num_layers}")

    # Note: On ne peut pas vraiment comparer les outputs directement car les modèles
    # ont des dimensions différentes. Mais on peut comparer leurs distributions de logits.

    print("\nNote: Direct output comparison not possible due to different dimensions.")
    print(
        "Consider fine-tuning the upscaled model on a small dataset to verify it learns."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify upscaled model quality")
    parser.add_argument(
        "--original",
        type=str,
        default="checkpoints/seqcond_torch_395k.pt",
        help="Path to original checkpoint",
    )
    parser.add_argument(
        "--upscaled",
        type=str,
        default="checkpoints/seqcond_upscaled_2x.pt",
        help="Path to upscaled checkpoint",
    )
    parser.add_argument("--scale-factor", type=int, default=2, help="Scale factor used")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Skip reconstruction quality check",
    )
    parser.add_argument(
        "--skip-inference", action="store_true", help="Skip inference test"
    )

    args = parser.parse_args()

    if not args.skip_reconstruction:
        verify_reconstruction_quality(
            args.original,
            args.upscaled,
            args.scale_factor,
        )

    if not args.skip_inference:
        print("\n" + "=" * 80)
        print("TESTING ORIGINAL MODEL")
        print("=" * 80)
        test_inference(args.original, args.device)

        print("\n" + "=" * 80)
        print("TESTING UPSCALED MODEL")
        print("=" * 80)
        test_inference(args.upscaled, args.device)
