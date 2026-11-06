$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install Python 3.12 64-bit from python.org, including the Python launcher, then reopen PowerShell.' }
& py -3.12 -c "import platform; assert platform.architecture()[0] == '64bit'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 64-bit was not found. Install it first.' }
& py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
$Python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $Python -m pip install numpy==2.3.5 opencv-python-headless==5.0.0.93
if ($LASTEXITCODE -ne 0) { throw 'Image dependencies installation failed.' }
& $Python run.py --source samples --output outputs/windows_preview --backend torch --provenance public_sample
if ($LASTEXITCODE -ne 0) { throw 'Preview failed. If output directory already exists, use a new output path.' }
Write-Host 'Windows PyTorch preview completed. For mandatory ExecuTorch evidence, run on the Raspberry Pi. WSL can be used for Linux development.'
