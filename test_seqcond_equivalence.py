import jax
import jax.numpy as jnp
import os
import sys

# Ensure we can import the module from the current directory
# This assumes the script is run from the root of the repo (where 'seqcond' package is located)
# or from the 'seqcond' subdirectory.
if os.path.exists("seqcond/jax/seqcond_fast.py"):
    sys.path.append(os.getcwd())
elif os.path.exists("../seqcond/jax/seqcond_fast.py"):
    sys.path.append(os.path.dirname(os.getcwd()))

try:
    from seqcond.jax.seqcond_fast import SeqCondAttention
except ImportError:
    # Fallback if running directly inside the package
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from seqcond.jax.seqcond_fast import SeqCondAttention

def test_seqcond_equivalence():
    print("=" * 60)
    print("SeqCond Equivalence Test (Matrix vs Scan)")
    print("=" * 60)
    
    # Check Devices
    devices = jax.devices()
    print(f"JAX Devices: {devices}")
    print(f"Platform: {jax.lib.xla_bridge.get_backend().platform}")
    
    # Config
    B, L, D = 2, 256, 64 
    print(f"\nConfiguration: Batch={B}, Length={L}, Dim={D}")
    
    # Data
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (B, L, D), dtype=jnp.bfloat16).astype(jnp.float32)
    print(f"Input shape: {x.shape}")

    # 1. Run Matrix Path (use_square_matrix=True)
    print("\n[1/2] Running Matrix Path (O(L^2))...")
    model_matrix = SeqCondAttention(
        num_heads=4, 
        num_query_heads=2, 
        expand_factor=1,
        use_square_matrix=True,
        # Ensure consistent dtypes
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    # Initialize variables
    variables = model_matrix.init(key, x)
    
    # Run Matrix Path
    # JIT compile to ensure it runs as expected on device
    apply_fn = jax.jit(model_matrix.apply)
    out_matrix = apply_fn(variables, x)
    
    # 2. Run Scan Path (use_square_matrix=False)
    print("[2/2] Running Scan Path (O(L))...")
    model_scan = SeqCondAttention(
        num_heads=4, 
        num_query_heads=2, 
        expand_factor=1,
        use_square_matrix=False,
        # Ensure consistent dtypes
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    # Run Scan Path using the SAME variables
    apply_fn_scan = jax.jit(model_scan.apply)
    out_scan = apply_fn_scan(variables, x)
    
    # 3. Comparaison
    print("\n[3/3] Comparing results...")
    
    # Cast to float32 for precise comparison if they were bfloat16
    out_matrix_f32 = out_matrix.astype(jnp.float32)
    out_scan_f32 = out_scan.astype(jnp.float32)
    
    diff = jnp.abs(out_matrix_f32 - out_scan_f32)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)
    
    print(f"Max Difference:  {max_diff:.6e}")
    print(f"Mean Difference: {mean_diff:.6e}")
    
    # Loose tolerance for BF16 operations even if compute is float32 due to hardware
    tolerance = 1e-4 
    
    if max_diff < tolerance: 
        print(f"\n✅ SUCCESS: Matrix and Scan paths are equivalent (diff < {tolerance})!")
    else:
        print(f"\n❌ FAIL: Paths diverge (diff >= {tolerance}).")
        
        # Debug info
        print("\nDebug Indices of Max Diff:")
        idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        print(f"Index: {idx}")
        print(f"Matrix val: {out_matrix_f32[idx]}")
        print(f"Scan val:   {out_scan_f32[idx]}")

if __name__ == "__main__":
    test_seqcond_equivalence()