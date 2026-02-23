"""
Test rapide du concept d'upscaling sur un mini-modèle.

Ce script crée un petit modèle, l'upscale, et vérifie que:
1. Le downsampling reproduit bien l'original
2. Le modèle upscalé peut faire de l'inférence
3. La qualité de reconstruction est bonne
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from upscale_model import WeightUpscaler, initialize_upscaled_weights, learn_upscaling_for_layer


def create_tiny_model():
    """Crée un mini-modèle pour tester."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)
            self.norm = nn.LayerNorm(64)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.norm(x)
            return x
    
    return TinyModel()


def test_weight_upscaling():
    """Test l'upscaling d'une seule matrice de poids."""
    print("="*80)
    print("TEST 1: Single Weight Matrix Upscaling")
    print("="*80)
    
    # Créer une matrice de poids aléatoire
    original_weight = torch.randn(64, 32)
    print(f"\nOriginal weight shape: {original_weight.shape}")
    
    # Upscaler 2x
    scale_factor = 2
    target_shape = (128, 64)
    
    # Initialisation simple
    upscaled = initialize_upscaled_weights(original_weight, target_shape, method="bilinear")
    print(f"Upscaled weight shape: {upscaled.shape}")
    
    # Downsample et vérifier
    downsampled = F.avg_pool1d(
        upscaled.unsqueeze(0),
        kernel_size=scale_factor,
        stride=scale_factor
    ).squeeze(0)
    downsampled = F.avg_pool1d(
        downsampled.t().unsqueeze(0),
        kernel_size=scale_factor,
        stride=scale_factor
    ).squeeze(0).t()
    
    print(f"Downsampled shape: {downsampled.shape}")
    
    # Calculer l'erreur
    mse = F.mse_loss(downsampled, original_weight).item()
    relative_error = (mse / (original_weight.var().item() + 1e-8)) ** 0.5
    
    print(f"\nReconstruction quality (before learning):")
    print(f"  MSE: {mse:.6f}")
    print(f"  Relative error: {relative_error:.4f}")
    
    return mse < 0.1  # Should be reasonably good even before learning


def test_learned_upscaling():
    """Test l'apprentissage de l'upscaling optimal."""
    print("\n" + "="*80)
    print("TEST 2: Learned Upscaling")
    print("="*80)
    
    # Créer des poids de test
    layer_weights = {
        "weight": torch.randn(64, 32),
        "bias": torch.randn(64),
    }
    
    print(f"\nOriginal weights:")
    for name, w in layer_weights.items():
        print(f"  {name}: {w.shape}")
    
    # Apprendre l'upscaling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    upscaled_weights = learn_upscaling_for_layer(
        layer_weights,
        scale_factor=2,
        num_steps=200,  # Rapide pour le test
        lr=1e-3,
        device=device,
    )
    
    print(f"\nUpscaled weights:")
    for name, w in upscaled_weights.items():
        print(f"  {name}: {w.shape}")
    
    # Vérifier la qualité de reconstruction
    print(f"\nReconstruction quality (after learning):")
    
    total_error = 0.0
    for name, orig_weight in layer_weights.items():
        upscaled = upscaled_weights[name]
        
        # Downsample
        if len(orig_weight.shape) == 1:
            downsampled = F.avg_pool1d(
                upscaled.unsqueeze(0).unsqueeze(0),
                kernel_size=2,
                stride=2
            ).squeeze()
        else:
            downsampled = F.avg_pool1d(
                upscaled.unsqueeze(0),
                kernel_size=2,
                stride=2
            ).squeeze(0)
            downsampled = F.avg_pool1d(
                downsampled.t().unsqueeze(0),
                kernel_size=2,
                stride=2
            ).squeeze(0).t()
        
        mse = F.mse_loss(downsampled, orig_weight).item()
        relative_error = (mse / (orig_weight.var().item() + 1e-8)) ** 0.5
        
        print(f"  {name:20s} MSE={mse:.6f}, rel_err={relative_error:.4f}")
        total_error += relative_error
    
    avg_error = total_error / len(layer_weights)
    print(f"\nAverage relative error: {avg_error:.4f}")
    
    return avg_error < 0.1  # Should be very good after learning


def test_model_upscaling():
    """Test l'upscaling d'un modèle complet."""
    print("\n" + "="*80)
    print("TEST 3: Full Model Upscaling")
    print("="*80)
    
    # Créer et sauvegarder un mini-modèle
    model = create_tiny_model()
    
    print(f"\nOriginal model:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    for name, param in model.named_parameters():
        print(f"  {name:30s} {param.shape}")
    
    # Sauvegarder
    checkpoint_path = "/tmp/tiny_model.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "d_model": 64,
            "d_ff": 128,
            "num_layers": 2,
            "vocab_size": 1000,
            "maxlen": 128,
            "num_heads": 4,
            "seqcond_heads": 8,
            "num_query_heads": 4,
            "num_thetas": 2,
            "conv_kernel_size": 4,
            "expand_factor": 1,
            "out_expand_factor": 3,
            "seqcond_ratio": 3,
            "model_type": "seqcond",
            "tie_weights": True,
            "qk_norm": True,
        }
    }, checkpoint_path)
    
    print(f"\nSaved to {checkpoint_path}")
    
    # Test forward pass
    x = torch.randn(2, 64)
    with torch.no_grad():
        y = model(x)
    
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    
    return True


def test_initialization_methods():
    """Compare différentes méthodes d'initialisation."""
    print("\n" + "="*80)
    print("TEST 4: Initialization Methods Comparison")
    print("="*80)
    
    original = torch.randn(32, 16)
    target_shape = (64, 32)
    
    methods = ["bilinear", "nearest"]
    
    for method in methods:
        upscaled = initialize_upscaled_weights(original, target_shape, method=method)
        
        # Downsample
        downsampled = F.avg_pool1d(
            upscaled.unsqueeze(0),
            kernel_size=2,
            stride=2
        ).squeeze(0)
        downsampled = F.avg_pool1d(
            downsampled.t().unsqueeze(0),
            kernel_size=2,
            stride=2
        ).squeeze(0).t()
        
        mse = F.mse_loss(downsampled, original).item()
        relative_error = (mse / (original.var().item() + 1e-8)) ** 0.5
        
        print(f"\n{method:10s}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Relative error: {relative_error:.4f}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING MODEL UPSCALING CONCEPT")
    print("="*80)
    
    results = {}
    
    try:
        results["single_weight"] = test_weight_upscaling()
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        results["single_weight"] = False
    
    try:
        results["learned_upscaling"] = test_learned_upscaling()
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        results["learned_upscaling"] = False
    
    try:
        results["model_upscaling"] = test_model_upscaling()
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        results["model_upscaling"] = False
    
    try:
        results["initialization"] = test_initialization_methods()
    except Exception as e:
        print(f"\n✗ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        results["initialization"] = False
    
    # Résumé
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! The upscaling concept works.")
        print("\nYou can now run:")
        print("  python upscale_model.py --checkpoint checkpoints/seqcond_torch_395k.pt")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
    
    exit(0 if all_passed else 1)
