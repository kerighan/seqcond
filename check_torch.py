import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is NOT available - CPU-only installation")
except ImportError as e:
    print("PyTorch is NOT installed:", e)
