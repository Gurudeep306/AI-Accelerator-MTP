#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon). Run from the project folder:
#   bash demo/setup_mac.sh
set -e
cd "$(dirname "$0")/.."
PY=$(command -v python3.11 || command -v python3.12 || command -v python3)
echo "using $PY ($($PY --version))"
$PY -c 'import sys; assert sys.version_info >= (3, 10), "need Python 3.10 or newer: brew install python@3.11"'
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt || pip install torch numpy "git+https://github.com/scalesim-project/SCALE-Sim.git@9f98c4371055a54c75209c2e02b640b897550532"
python3 - <<'PY'
import sys; sys.path.insert(0, 'src')
from sim_config import patch_scalesim
patch_scalesim()
import torch, scalesim, importlib.metadata as m
print('torch', torch.__version__, '| scalesim', m.version('scalesim'))
PY
echo "setup done. Next:  source .venv/bin/activate && bash demo/demo.sh"
