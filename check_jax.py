import os
import time
import jax

print(f"--- JAX Diagnostic (PID: {os.getpid()}) ---")

# 1. Check Environment
print("\n[Environment Variables]")
keys = ['TPU_NAME', 'TPU_IP_ADDRESS', 'TPU_CHIPS_PER_HOST_BOUNDS', 'WORKER_ID', 'JAX_PLATFORMS', 'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']
for key in keys:
    print(f"{key}: {os.environ.get(key, 'Not Set')}")

# 2. Initialize
print("\n[Initialization]")
print("Calling jax.distributed.initialize()... (This may hang if other workers are not running)")
start = time.time()

try:
    # Tente l'initialisation
    jax.distributed.initialize()
    duration = time.time() - start
    print(f"SUCCESS! Initialized in {duration:.2f}s")
    
    # 3. Topology Info
    print("\n[Topology]")
    print(f"Process Index (Rank): {jax.process_index()}")
    print(f"Total Processes: {jax.process_count()}")
    print(f"Local Devices: {len(jax.local_devices())} -> {jax.local_devices()}")
    print(f"Global Devices: {jax.device_count()}")
    
except KeyboardInterrupt:
    print(f"\nINTERRUPTED by user after {time.time() - start:.2f}s")
except Exception as e:
    print(f"\nFAILED with error: {e}")
