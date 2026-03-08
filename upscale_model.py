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
        Supporte les tenseurs de 1D à N-D en appliquant une interpolation
        dimension par dimension pour les dimensions qui changent de taille.
        """
        ndim = len(weights.shape)

        if ndim == 1:
            w = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            downsampled = F.interpolate(
                w, size=self.input_shape[0], mode="linear", align_corners=True
            ).squeeze()
            return downsampled

        elif ndim == 2:
            w = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            downsampled = (
                F.interpolate(
                    w, size=self.input_shape, mode="bilinear", align_corners=True
                )
                .squeeze(0)
                .squeeze(0)
            )
            return downsampled

        else:
            # Generic N-D: downsample each dimension that changed size
            # by reshaping to 2D, interpolating, and reshaping back.
            result = weights
            current_shape = list(weights.shape)
            for dim in range(ndim):
                if current_shape[dim] == self.input_shape[dim]:
                    continue
                # Flatten all other dims into one, keeping target dim last
                # Move target dim to last position
                perm = list(range(ndim))
                perm.remove(dim)
                perm.append(dim)
                result = result.permute(*perm)  # (..., target_dim)
                flat_shape = result.shape
                result = result.reshape(-1, flat_shape[-1])  # (N, target_dim)
                result = result.unsqueeze(1)  # (N, 1, target_dim)
                result = F.interpolate(
                    result,
                    size=self.input_shape[dim],
                    mode="linear",
                    align_corners=True,
                ).squeeze(
                    1
                )  # (N, new_target_dim)
                # Reshape back and un-permute
                new_flat_shape = list(flat_shape[:-1]) + [self.input_shape[dim]]
                result = result.reshape(new_flat_shape)
                inv_perm = [0] * ndim
                for i, p in enumerate(perm):
                    inv_perm[p] = i
                result = result.permute(*inv_perm)
                current_shape[dim] = self.input_shape[dim]
            return result


def initialize_upscaled_weights(
    original_weights: torch.Tensor,
    target_shape: Tuple[int, ...],
    method: str = "bilinear",
) -> torch.Tensor:
    """
    Initialise les poids upscalés avec une interpolation simple.
    Supporte les tenseurs de 1D à N-D.
    """
    ndim = len(original_weights.shape)

    if ndim == 1:
        indices = torch.linspace(0, original_weights.shape[0] - 1, target_shape[0])
        indices_low = indices.long()
        indices_high = (indices_low + 1).clamp(max=original_weights.shape[0] - 1)
        alpha = indices - indices_low.float()
        upscaled = (1 - alpha) * original_weights[
            indices_low
        ] + alpha * original_weights[indices_high]
        return upscaled

    elif ndim == 2:
        w = original_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        upscaled = F.interpolate(
            w,
            size=target_shape,
            mode="bilinear",
            align_corners=True,
        )
        return upscaled.squeeze(0).squeeze(0)

    else:
        # Generic N-D: interpolate each changed dimension sequentially
        result = original_weights
        current_shape = list(original_weights.shape)
        for dim in range(ndim):
            if current_shape[dim] == target_shape[dim]:
                continue
            src_size = current_shape[dim]
            tgt_size = target_shape[dim]
            # Move target dim to last, flatten others, interpolate, reshape back
            perm = list(range(len(current_shape)))
            perm.remove(dim)
            perm.append(dim)
            result = result.permute(*perm)
            flat_shape = result.shape
            result = result.reshape(-1, flat_shape[-1]).unsqueeze(1)  # (N, 1, src)
            result = F.interpolate(
                result,
                size=tgt_size,
                mode="linear",
                align_corners=True,
            ).squeeze(
                1
            )  # (N, tgt)
            new_flat_shape = list(flat_shape[:-1]) + [tgt_size]
            result = result.reshape(new_flat_shape)
            inv_perm = [0] * len(current_shape)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            result = result.permute(*inv_perm)
            current_shape[dim] = tgt_size
        # Add small noise to break symmetry
        result = result + torch.randn_like(result) * 0.001 * (result.std() + 1e-8)
        return result


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


def compute_target_shapes(
    source_state_dict: Dict[str, torch.Tensor],
    target_config: ModelConfig,
) -> Dict[str, Tuple[int, ...]]:
    """
    Compute exact target shapes by instantiating the target model.

    This is the most reliable approach: instead of guessing shapes from
    parameter names, we create the target model and read its state_dict shapes.
    """
    from seqcond.torch.model import SeqCondModel

    print("  Instantiating target model to compute exact shapes...")
    target_model = SeqCondModel(
        d_model=target_config.d_model,
        d_ff=target_config.d_ff,
        num_layers=target_config.num_layers,
        vocab_size=target_config.vocab_size,
        maxlen=target_config.maxlen,
        num_heads=target_config.num_heads,
        num_kv_heads=target_config.num_kv_heads,
        qk_norm=target_config.qk_norm,
        seqcond_heads=target_config.seqcond_heads,
        num_query_heads=target_config.num_query_heads,
        num_thetas=target_config.num_thetas,
        conv_kernel_size=target_config.conv_kernel_size,
        expand_factor=target_config.expand_factor,
        out_expand_factor=target_config.out_expand_factor,
        seqcond_ratio=target_config.seqcond_ratio,
        dropout=0.0,
    )

    target_state_dict = target_model.state_dict()
    target_shapes = {}

    matched = 0
    unmatched = []
    for name in source_state_dict:
        if name in target_state_dict:
            target_shapes[name] = tuple(target_state_dict[name].shape)
            matched += 1
        else:
            # Keep original shape for params not in target model
            target_shapes[name] = tuple(source_state_dict[name].shape)
            unmatched.append(name)

    if unmatched:
        print(f"  Warning: {len(unmatched)} source params not found in target model:")
        for n in unmatched[:5]:
            print(f"    {n}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    print(f"  Matched {matched}/{len(source_state_dict)} parameters")

    # Free target model memory
    del target_model, target_state_dict

    return target_shapes


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extraire la config et les poids
    if "config" in checkpoint:
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        config_dict = {
            k: v for k, v in checkpoint["config"].items() if k in valid_fields
        }
        original_config = ModelConfig(**config_dict)
    else:
        print("No config found in checkpoint, using default large config")
        original_config = ModelConfig.large()

    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

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

    # Compute exact target shapes by instantiating target model
    target_shapes = compute_target_shapes(state_dict, target_config)

    # Grouper les poids par layer
    layer_groups = {}
    for name, param in state_dict.items():
        parts = name.split(".")
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

    # Count how many params will change
    n_changed = sum(
        1 for name, p in state_dict.items() if tuple(p.shape) != target_shapes[name]
    )
    print(f"Parameters to upscale: {n_changed}/{len(state_dict)}")

    # Upscaler chaque groupe de layers
    upscaled_state_dict = {}

    for layer_name, layer_weights in tqdm(
        layer_groups.items(), desc="Upscaling layers"
    ):
        # Gather target shapes for this layer
        layer_target_shapes = {name: target_shapes[name] for name in layer_weights}

        # Count changes in this layer
        changes = [
            name
            for name, w in layer_weights.items()
            if tuple(w.shape) != layer_target_shapes[name]
        ]
        if changes:
            print(
                f"\nProcessing {layer_name} ({len(layer_weights)} params, {len(changes)} to upscale)"
            )
            for name in changes:
                print(
                    f"  {name}: {tuple(layer_weights[name].shape)} -> {layer_target_shapes[name]}"
                )
        else:
            print(
                f"\nProcessing {layer_name} ({len(layer_weights)} params, no changes)"
            )

        # Apprendre l'upscaling pour cette layer
        upscaled_layer = learn_upscaling_for_layer(
            layer_weights,
            target_shapes=layer_target_shapes,
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
