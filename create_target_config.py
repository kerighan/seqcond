"""
Helper script pour créer une config target pour l'upscaling.

Usage:
    python create_target_config.py --checkpoint checkpoints/seqcond_torch_395k.pt --output target_config.json
"""

import torch
import json
import argparse
from seqcond.config import ModelConfig


def create_target_config_interactive(original_config: ModelConfig) -> ModelConfig:
    """Crée une config target de manière interactive."""
    print("\n" + "=" * 80)
    print("CRÉATION DE LA CONFIG TARGET")
    print("=" * 80)

    print("\nConfig originale:")
    print(f"  d_model: {original_config.d_model}")
    print(f"  d_ff: {original_config.d_ff}")
    print(f"  num_layers: {original_config.num_layers}")
    print(f"  num_heads: {original_config.num_heads}")
    print(f"  seqcond_heads: {original_config.seqcond_heads}")
    print(f"  num_query_heads: {original_config.num_query_heads}")
    print(f"  num_kv_heads: {original_config.num_kv_heads}")

    print(
        "\nEntrez les nouvelles valeurs (appuyez sur Entrée pour garder la valeur originale):"
    )

    def get_int_input(prompt: str, default: int) -> int:
        while True:
            val = input(f"{prompt} [{default}]: ").strip()
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                print("Erreur: entrez un nombre entier")

    d_model = get_int_input("d_model", original_config.d_model)
    d_ff = get_int_input("d_ff", original_config.d_ff)
    num_layers = get_int_input("num_layers", original_config.num_layers)
    num_heads = get_int_input("num_heads", original_config.num_heads)
    seqcond_heads = get_int_input("seqcond_heads", original_config.seqcond_heads)
    num_query_heads = get_int_input("num_query_heads", original_config.num_query_heads)
    num_kv_heads = get_int_input(
        "num_kv_heads", original_config.num_kv_heads or original_config.num_heads
    )

    target_config = ModelConfig(
        model_type=original_config.model_type,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        vocab_size=original_config.vocab_size,
        maxlen=original_config.maxlen,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seqcond_heads=seqcond_heads,
        num_query_heads=num_query_heads,
        num_thetas=original_config.num_thetas,
        conv_kernel_size=original_config.conv_kernel_size,
        expand_factor=original_config.expand_factor,
        out_expand_factor=original_config.out_expand_factor,
        seqcond_ratio=original_config.seqcond_ratio,
        dropout=original_config.dropout,
        tie_weights=original_config.tie_weights,
        qk_norm=original_config.qk_norm,
    )

    return target_config


def create_target_config_preset(
    original_config: ModelConfig, preset: str
) -> ModelConfig:
    """Crée une config target basée sur un preset."""

    if preset == "2x":
        scale = 2
    elif preset == "1.5x":
        scale = 1.5
    elif preset == "3x":
        scale = 3
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return ModelConfig(
        model_type=original_config.model_type,
        d_model=int(original_config.d_model * scale),
        d_ff=int(original_config.d_ff * scale),
        num_layers=original_config.num_layers,
        vocab_size=original_config.vocab_size,
        maxlen=original_config.maxlen,
        num_heads=int(original_config.num_heads * scale),
        num_kv_heads=int(
            (original_config.num_kv_heads or original_config.num_heads) * scale
        ),
        seqcond_heads=int(original_config.seqcond_heads * scale),
        num_query_heads=int(original_config.num_query_heads * scale),
        num_thetas=original_config.num_thetas,
        conv_kernel_size=original_config.conv_kernel_size,
        expand_factor=original_config.expand_factor,
        out_expand_factor=original_config.out_expand_factor,
        seqcond_ratio=original_config.seqcond_ratio,
        dropout=original_config.dropout,
        tie_weights=original_config.tie_weights,
        qk_norm=original_config.qk_norm,
    )


def estimate_params(config: ModelConfig) -> int:
    """Estime le nombre de paramètres d'un modèle."""
    # Approximation grossière
    vocab_size = config.vocab_size
    d_model = config.d_model
    d_ff = config.d_ff
    num_layers = config.num_layers

    # Embedding
    params = vocab_size * d_model

    # Layers (approximation)
    params_per_layer = (
        # Attention/SeqCond
        d_model * d_model * 4  # Q, K, V, O projections
        +
        # FFN
        d_model * d_ff * 2
        +
        # Norms
        d_model * 2
    )

    params += params_per_layer * num_layers

    # Output head (tied with embedding)
    # params += vocab_size * d_model

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create target config for model upscaling"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to original checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="target_config.json",
        help="Path to save target config JSON",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["2x", "1.5x", "3x", "interactive"],
        default="interactive",
        help="Preset scaling factor or interactive mode",
    )

    args = parser.parse_args()

    # Charger le checkpoint original
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "config" in checkpoint:
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        filtered = {k: v for k, v in checkpoint["config"].items() if k in valid_fields}
        original_config = ModelConfig(**filtered)
    else:
        print("No config found in checkpoint, using default large config")
        original_config = ModelConfig.large()

    # Créer la target config
    if args.preset == "interactive":
        target_config = create_target_config_interactive(original_config)
    else:
        target_config = create_target_config_preset(original_config, args.preset)

    # Afficher un résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    orig_params = estimate_params(original_config)
    target_params = estimate_params(target_config)

    print(f"\nOriginal model:")
    print(f"  d_model: {original_config.d_model}")
    print(f"  d_ff: {original_config.d_ff}")
    print(f"  num_layers: {original_config.num_layers}")
    print(f"  num_heads: {original_config.num_heads}")
    print(f"  seqcond_heads: {original_config.seqcond_heads}")
    print(f"  Estimated params: ~{orig_params/1e6:.1f}M")

    print(f"\nTarget model:")
    print(
        f"  d_model: {target_config.d_model} ({target_config.d_model/original_config.d_model:.2f}x)"
    )
    print(
        f"  d_ff: {target_config.d_ff} ({target_config.d_ff/original_config.d_ff:.2f}x)"
    )
    print(f"  num_layers: {target_config.num_layers}")
    print(
        f"  num_heads: {target_config.num_heads} ({target_config.num_heads/original_config.num_heads:.2f}x)"
    )
    print(
        f"  seqcond_heads: {target_config.seqcond_heads} ({target_config.seqcond_heads/original_config.seqcond_heads:.2f}x)"
    )
    print(f"  Estimated params: ~{target_params/1e6:.1f}M")
    print(f"  Scaling ratio: {target_params/orig_params:.2f}x")

    # Sauvegarder
    with open(args.output, "w") as f:
        json.dump(target_config.to_dict(), f, indent=2)

    print(f"\n✓ Target config saved to {args.output}")
    print(f"\nYou can now run:")
    print(f"  python upscale_model.py \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --target-config {args.output} \\")
    print(f"    --output checkpoints/seqcond_upscaled.pt")
