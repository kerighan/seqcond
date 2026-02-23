"""
Model Upscaling via Learned Deconvolution

Cette approche permet d'upscaler un modèle pré-entraîné en apprenant des transformations
de déconvolution layer-by-layer qui préservent les connaissances du modèle original.

Concept:
1. Pour chaque layer, on upscale les poids (ex: d_model 1024 -> 2048)
2. On apprend une déconvolution qui, après downsampling, reproduit les poids originaux
3. On assemble toutes les layers upscalées pour créer un nouveau modèle 2x plus gros

Avantages:
- Réutilise le compute déjà investi dans le modèle 370M
- Fournit une meilleure initialisation qu'un random init
- Permet de scale up avec un budget compute limité
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import json

from seqcond.config import ModelConfig
from seqcond.torch.model import SeqCondModel


class WeightUpscaler(nn.Module):
    """
    Apprend une transformation de déconvolution pour upscaler des poids.

    L'objectif est d'apprendre une transformation W_small -> W_large telle que
    downsample(W_large) ≈ W_small

    Supporte des scaling factors différents par dimension (ex: d_model x2, num_heads x1.5)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        upscale_method: str = "deconv",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.upscale_method = upscale_method

        # Calculer les facteurs de scaling (peuvent être non-entiers)
        self.scale_factors = tuple(o / i for o, i in zip(output_shape, input_shape))

        # Initialiser les poids upscalés avec une interpolation simple
        self.upscaled_weights = nn.Parameter(torch.zeros(output_shape))

    def forward(self, target_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: retourne les poids upscalés.
        La loss sera calculée en comparant downsample(upscaled) avec target.
        """
        return self.upscaled_weights

    def downsample(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Downsample les poids pour comparer avec l'original.
        Utilise interpolation adaptative pour gérer des scaling factors non-entiers.
        """
        if len(weights.shape) == 1:
            # Vecteur (bias, norm weights, etc.)
            # Utiliser interpolation pour supporter des ratios non-entiers
            w = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            downsampled = F.interpolate(
                w, size=self.input_shape[0], mode="linear", align_corners=True
            ).squeeze()
            return downsampled

        elif len(weights.shape) == 2:
            # Matrice (Linear layers)
            # Interpolation 2D pour supporter des ratios non-entiers
            w = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            downsampled = (
                F.interpolate(
                    w, size=self.input_shape, mode="bilinear", align_corners=True
                )
                .squeeze(0)
                .squeeze(0)
            )
            return downsampled

        elif len(weights.shape) == 3:
            # Conv1D ou autre (features, kernel, channels)
            # On ne scale que la dimension des features (dim 0)
            # Les autres dimensions restent identiques
            if weights.shape[1:] != self.input_shape[1:]:
                raise ValueError(
                    f"Conv shape mismatch: {weights.shape[1:]} vs {self.input_shape[1:]}"
                )

            # Interpolation sur la dimension 0 seulement
            w = weights.transpose(0, 2).unsqueeze(0)  # (1, channels, kernel, features)
            downsampled = (
                F.interpolate(
                    w,
                    size=(weights.shape[1], self.input_shape[0]),
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(0)
                .transpose(0, 2)
            )  # Back to (features, kernel, channels)
            return downsampled

        else:
            raise NotImplementedError(
                f"Downsampling for shape {weights.shape} not implemented"
            )


def initialize_upscaled_weights(
    original_weights: torch.Tensor,
    target_shape: Tuple[int, ...],
    method: str = "bilinear",
) -> torch.Tensor:
    """
    Initialise les poids upscalés avec une interpolation simple.
    """
    if len(original_weights.shape) == 1:
        # Vecteur: interpolation linéaire
        scale = target_shape[0] / original_weights.shape[0]
        indices = torch.linspace(0, original_weights.shape[0] - 1, target_shape[0])
        indices_low = indices.long()
        indices_high = (indices_low + 1).clamp(max=original_weights.shape[0] - 1)
        alpha = indices - indices_low.float()

        upscaled = (1 - alpha) * original_weights[
            indices_low
        ] + alpha * original_weights[indices_high]
        return upscaled

    elif len(original_weights.shape) == 2:
        # Matrice: interpolation bilinéaire
        w = original_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        upscaled = F.interpolate(
            w,
            size=target_shape,
            mode="bilinear" if method == "bilinear" else "nearest",
            align_corners=True if method == "bilinear" else None,
        )
        return upscaled.squeeze(0).squeeze(0)

    elif len(original_weights.shape) == 3:
        # Conv weights: on scale seulement la première dimension
        scale = target_shape[0] // original_weights.shape[0]
        # Répéter et ajouter du bruit pour diversifier
        upscaled = original_weights.repeat_interleave(scale, dim=0)
        # Ajouter un petit bruit gaussien pour briser la symétrie
        upscaled = upscaled + torch.randn_like(upscaled) * 0.01 * upscaled.std()
        return upscaled

    else:
        raise NotImplementedError(
            f"Initialization for shape {original_weights.shape} not implemented"
        )


def learn_upscaling_for_layer(
    layer_weights: Dict[str, torch.Tensor],
    target_shapes: Dict[str, Tuple[int, ...]],
    num_steps: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Apprend l'upscaling optimal pour une layer donnée.

    Args:
        layer_weights: Dict des poids de la layer {name: tensor}
        target_shapes: Dict des target shapes {name: shape}
        num_steps: Nombre d'étapes d'optimisation
        lr: Learning rate
        device: Device pour l'entraînement

    Returns:
        Dict des poids upscalés
    """
    upscalers = {}
    optimizers = []

    # Créer un upscaler pour chaque poids
    for name, weight in layer_weights.items():
        original_shape = weight.shape

        # Récupérer la target shape
        if name not in target_shapes:
            print(f"Warning: No target shape for {name}, keeping original shape")
            target_shape = original_shape
        else:
            target_shape = target_shapes[name]

        # Skip si pas de changement
        if original_shape == target_shape:
            continue

        # Créer l'upscaler
        upscaler = WeightUpscaler(original_shape, target_shape).to(device)

        # Initialiser avec interpolation
        with torch.no_grad():
            upscaler.upscaled_weights.data = initialize_upscaled_weights(
                weight.to(device), target_shape
            )

        upscalers[name] = upscaler
        optimizers.append(torch.optim.Adam(upscaler.parameters(), lr=lr))

    # Optimiser tous les upscalers ensemble
    pbar = tqdm(range(num_steps), desc="Learning upscaling")
    for step in pbar:
        total_loss = 0.0

        for (name, weight), upscaler, optimizer in zip(
            layer_weights.items(), upscalers.values(), optimizers
        ):
            optimizer.zero_grad()

            # Forward: obtenir les poids upscalés
            upscaled = upscaler(weight.to(device))

            # Downsample et comparer avec l'original
            downsampled = upscaler.downsample(upscaled)

            # Loss: reconstruction + régularisation
            recon_loss = F.mse_loss(downsampled, weight.to(device))

            # Régularisation: encourager la smoothness
            smooth_loss = 0.0
            if len(upscaled.shape) == 2:
                # Pour les matrices, pénaliser les grandes variations
                smooth_loss = (
                    torch.mean((upscaled[1:] - upscaled[:-1]) ** 2)
                    + torch.mean((upscaled[:, 1:] - upscaled[:, :-1]) ** 2)
                ) * 0.01

            loss = recon_loss + smooth_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{total_loss:.6f}"})

    # Extraire les poids upscalés finaux
    upscaled_weights = {}
    for name, upscaler in upscalers.items():
        upscaled_weights[name] = upscaler.upscaled_weights.data.cpu()

    return upscaled_weights


def compute_dimension_mapping(
    original_config: ModelConfig,
    target_config: ModelConfig,
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Calcule le mapping des dimensions entre le modèle original et target.

    Returns:
        Dict avec le mapping pour chaque type de paramètre:
        {
            'embedding': {'d_model': (orig, target)},
            'linear_d_model': {'in': (orig, target), 'out': (orig, target)},
            'seqcond': {...},
            etc.
        }
    """
    mapping = {
        "d_model": (original_config.d_model, target_config.d_model),
        "d_ff": (original_config.d_ff, target_config.d_ff),
        "num_heads": (original_config.num_heads, target_config.num_heads),
        "seqcond_heads": (original_config.seqcond_heads, target_config.seqcond_heads),
        "num_query_heads": (
            original_config.num_query_heads,
            target_config.num_query_heads,
        ),
        "num_kv_heads": (
            original_config.num_kv_heads or original_config.num_heads,
            target_config.num_kv_heads or target_config.num_heads,
        ),
        "vocab_size": (original_config.vocab_size, target_config.vocab_size),
    }

    return mapping


def infer_target_shape(
    param_name: str,
    original_shape: Tuple[int, ...],
    dim_mapping: Dict[str, Tuple[int, int]],
    original_config: ModelConfig,
    target_config: ModelConfig,
) -> Tuple[int, ...]:
    """
    Infère la shape target pour un paramètre donné basé sur son nom et shape originale.

    Args:
        param_name: Nom du paramètre (ex: 'blocks.0.attention.q_proj.weight')
        original_shape: Shape originale du paramètre
        dim_mapping: Mapping des dimensions calculé par compute_dimension_mapping
        original_config: Config du modèle original
        target_config: Config du modèle target

    Returns:
        Target shape pour ce paramètre
    """
    # Embedding layers: (vocab_size, d_model)
    if "embedding" in param_name or "lm_head" in param_name:
        if len(original_shape) == 2:
            return (dim_mapping["vocab_size"][1], dim_mapping["d_model"][1])
        elif len(original_shape) == 1:
            return (dim_mapping["vocab_size"][1],)

    # Norm layers: (d_model,)
    if "norm" in param_name and len(original_shape) == 1:
        return (dim_mapping["d_model"][1],)

    # Attention/Linear layers dans transformer blocks
    if "transformer_block" in param_name:
        if "weight" in param_name and len(original_shape) == 2:
            # Déterminer si c'est d_model->d_ff, d_ff->d_model, ou d_model->d_model
            if original_shape[0] == original_config.d_ff:
                out_dim = dim_mapping["d_ff"][1]
            elif original_shape[0] == original_config.d_model:
                out_dim = dim_mapping["d_model"][1]
            else:
                # Proportionnel
                ratio = original_shape[0] / original_config.d_model
                out_dim = int(target_config.d_model * ratio)

            if original_shape[1] == original_config.d_ff:
                in_dim = dim_mapping["d_ff"][1]
            elif original_shape[1] == original_config.d_model:
                in_dim = dim_mapping["d_model"][1]
            else:
                ratio = original_shape[1] / original_config.d_model
                in_dim = int(target_config.d_model * ratio)

            return (out_dim, in_dim)
        elif "bias" in param_name and len(original_shape) == 1:
            if original_shape[0] == original_config.d_ff:
                return (dim_mapping["d_ff"][1],)
            elif original_shape[0] == original_config.d_model:
                return (dim_mapping["d_model"][1],)
            else:
                ratio = original_shape[0] / original_config.d_model
                return (int(target_config.d_model * ratio),)

    # SeqCond blocks
    if "seqcond_block" in param_name:
        if "weight" in param_name and len(original_shape) == 2:
            # Calculer les dimensions basées sur les formules de SeqCond
            ratio_out = original_shape[0] / original_config.d_model
            ratio_in = original_shape[1] / original_config.d_model
            out_dim = int(target_config.d_model * ratio_out)
            in_dim = int(target_config.d_model * ratio_in)
            return (out_dim, in_dim)
        elif "weight" in param_name and len(original_shape) == 3:
            # Conv weights: (features, kernel, channels)
            ratio = original_shape[0] / original_config.d_model
            return (
                int(target_config.d_model * ratio),
                original_shape[1],
                original_shape[2],
            )
        elif "bias" in param_name and len(original_shape) == 1:
            ratio = original_shape[0] / original_config.d_model
            return (int(target_config.d_model * ratio),)
        elif len(original_shape) == 1:
            # Autres vecteurs (theta_raw, etc.)
            # Garder la même taille si c'est lié à num_thetas ou autres hyperparams constants
            if original_shape[0] <= 10:  # Probablement un petit vecteur de config
                return original_shape
            ratio = original_shape[0] / original_config.d_model
            return (int(target_config.d_model * ratio),)

    # Par défaut: garder la même shape (pour les paramètres qu'on ne scale pas)
    return original_shape


def upscale_model(
    checkpoint_path: str,
    output_path: str,
    target_config: Optional[ModelConfig] = None,
    num_steps_per_layer: int = 500,
    device: str = "cuda",
):
    """
    Upscale un modèle complet en apprenant des déconvolutions layer-by-layer.

    Args:
        checkpoint_path: Chemin vers le checkpoint du modèle original
        output_path: Chemin pour sauvegarder le modèle upscalé
        target_config: Config du modèle target (si None, utilise 2x scaling uniforme)
        num_steps_per_layer: Nombre d'étapes d'optimisation par layer
        device: Device pour l'entraînement
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extraire la config et les poids
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        original_config = ModelConfig(**config_dict)
    else:
        print("No config found in checkpoint, using default large config")
        original_config = ModelConfig.large()

    state_dict = checkpoint.get("model", checkpoint)

    print(f"\nOriginal model config:")
    print(f"  d_model: {original_config.d_model}")
    print(f"  d_ff: {original_config.d_ff}")
    print(f"  num_layers: {original_config.num_layers}")
    print(f"  num_heads: {original_config.num_heads}")
    print(f"  seqcond_heads: {original_config.seqcond_heads}")
    print(f"  vocab_size: {original_config.vocab_size}")

    # Si pas de target_config, utiliser 2x scaling uniforme
    if target_config is None:
        print("\nNo target config provided, using 2x uniform scaling")
        target_config = ModelConfig(
            model_type=original_config.model_type,
            d_model=original_config.d_model * 2,
            d_ff=original_config.d_ff * 2,
            num_layers=original_config.num_layers,
            vocab_size=original_config.vocab_size,
            maxlen=original_config.maxlen,
            num_heads=original_config.num_heads * 2,
            num_kv_heads=(
                original_config.num_kv_heads * 2
                if original_config.num_kv_heads
                else None
            ),
            seqcond_heads=original_config.seqcond_heads * 2,
            num_query_heads=original_config.num_query_heads * 2,
            num_thetas=original_config.num_thetas,
            conv_kernel_size=original_config.conv_kernel_size,
            expand_factor=original_config.expand_factor,
            out_expand_factor=original_config.out_expand_factor,
            seqcond_ratio=original_config.seqcond_ratio,
            dropout=original_config.dropout,
            tie_weights=original_config.tie_weights,
            qk_norm=original_config.qk_norm,
        )

    print(f"\nTarget model config:")
    print(
        f"  d_model: {target_config.d_model} (ratio: {target_config.d_model/original_config.d_model:.2f}x)"
    )
    print(
        f"  d_ff: {target_config.d_ff} (ratio: {target_config.d_ff/original_config.d_ff:.2f}x)"
    )
    print(f"  num_layers: {target_config.num_layers}")
    print(
        f"  num_heads: {target_config.num_heads} (ratio: {target_config.num_heads/original_config.num_heads:.2f}x)"
    )
    print(
        f"  seqcond_heads: {target_config.seqcond_heads} (ratio: {target_config.seqcond_heads/original_config.seqcond_heads:.2f}x)"
    )

    # Calculer le mapping des dimensions
    dim_mapping = compute_dimension_mapping(original_config, target_config)

    # Grouper les poids par layer
    layer_groups = {}
    for name, param in state_dict.items():
        # Extraire le nom de la layer
        parts = name.split(".")

        # Identifier la layer (blocks.0, blocks.1, etc.)
        if "blocks" in parts:
            idx = parts.index("blocks")
            layer_name = ".".join(parts[: idx + 2])  # ex: "blocks.0"
        elif "embedding" in name:
            layer_name = "embedding"
        elif "lm_head" in name:
            layer_name = "lm_head"
        elif "final_norm" in name:
            layer_name = "final_norm"
        else:
            layer_name = "other"

        if layer_name not in layer_groups:
            layer_groups[layer_name] = {}

        layer_groups[layer_name][name] = param

    print(f"\nFound {len(layer_groups)} layer groups")

    # Upscaler chaque groupe de layers
    upscaled_state_dict = {}

    for layer_name, layer_weights in tqdm(
        layer_groups.items(), desc="Upscaling layers"
    ):
        print(f"\nProcessing {layer_name} ({len(layer_weights)} parameters)")

        # Calculer les target shapes pour chaque paramètre de cette layer
        target_shapes = {}
        for name, weight in layer_weights.items():
            target_shape = infer_target_shape(
                name, weight.shape, dim_mapping, original_config, target_config
            )
            target_shapes[name] = target_shape

            # Log si changement
            if weight.shape != target_shape:
                print(f"  {name}: {weight.shape} -> {target_shape}")

        # Apprendre l'upscaling pour cette layer
        upscaled_layer = learn_upscaling_for_layer(
            layer_weights,
            target_shapes=target_shapes,
            num_steps=num_steps_per_layer,
            lr=1e-3,
            device=device,
        )

        # Ajouter les poids upscalés
        upscaled_state_dict.update(upscaled_layer)

        # Ajouter les poids non-modifiés
        for name, weight in layer_weights.items():
            if name not in upscaled_state_dict:
                upscaled_state_dict[name] = weight

    # Sauvegarder le modèle upscalé
    print(f"\nSaving upscaled model to {output_path}")
    torch.save(
        {
            "model": upscaled_state_dict,
            "config": target_config.to_dict(),
            "original_checkpoint": checkpoint_path,
            "original_config": original_config.to_dict(),
        },
        output_path,
    )

    print("\nUpscaling complete!")

    # Calculer le nombre de paramètres
    original_params = sum(p.numel() for p in state_dict.values())
    upscaled_params = sum(p.numel() for p in upscaled_state_dict.values())

    print(f"\nParameter count:")
    print(f"  Original: {original_params:,} ({original_params/1e6:.1f}M)")
    print(f"  Upscaled: {upscaled_params:,} ({upscaled_params/1e6:.1f}M)")
    print(f"  Ratio: {upscaled_params/original_params:.2f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upscale a pretrained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seqcond_torch_395k.pt",
        help="Path to original checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/seqcond_upscaled_2x.pt",
        help="Path to save upscaled model",
    )
    parser.add_argument(
        "--target-config",
        type=str,
        default=None,
        help="Path to target config JSON file (if None, uses 2x uniform scaling)",
    )
    parser.add_argument(
        "--steps-per-layer", type=int, default=500, help="Optimization steps per layer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Charger la target config si fournie
    target_config = None
    if args.target_config:
        import json

        with open(args.target_config, "r") as f:
            target_config_dict = json.load(f)
        target_config = ModelConfig(**target_config_dict)
        print(f"Loaded target config from {args.target_config}")

    upscale_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        target_config=target_config,
        num_steps_per_layer=args.steps_per_layer,
        device=args.device,
    )
