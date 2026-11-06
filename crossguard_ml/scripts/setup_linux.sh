#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys,platform; assert (3,11)<=sys.version_info[:2]<=(3,13), "Use Python 3.11-3.13 (3.12 tested)"; assert platform.architecture()[0]=="64bit", "64-bit Python/OS required"'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# The CPU-only index avoids installing CUDA packages on a Linux PC.
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run.py --source samples --output outputs/sample_check --backend executorch --provenance public_sample
