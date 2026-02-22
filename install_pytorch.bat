@echo off
echo Uninstalling existing PyTorch...
pip uninstall torch torchvision torchaudio -y
echo.
echo Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo Installation complete!
echo.
echo Verifying installation...
python check_torch.py
pause
