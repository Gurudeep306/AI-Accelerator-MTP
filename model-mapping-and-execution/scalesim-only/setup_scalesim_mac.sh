#!/usr/bin/env bash
# One-time setup: SCALE-Sim v3 from its source repository, pinned to the commit the project used.
# Run inside scalesim-only/:   bash setup_scalesim_mac.sh
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.11 || command -v python3.12 || command -v python3)
$PY -c 'import sys; assert sys.version_info >= (3, 9), "need Python 3.9+: brew install python@3.11"'
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/scalesim-project/SCALE-Sim.git@9f98c4371055a54c75209c2e02b640b897550532"
python3 - <<'PY'
import scalesim, pathlib, shutil
p = pathlib.Path(scalesim.__file__).parent / 'memory' / 'double_buffered_scratchpad_mem.py'
s = p.read_text()
old, new = 'int(max(ofmap_serviced_cycles))', 'int(np.max(np.asarray(ofmap_serviced_cycles)))'
if old in s:
    shutil.copy(p, str(p) + '.orig'); p.write_text(s.replace(old, new)); print('patched SCALE-Sim (original kept as .orig)')
else:
    print('SCALE-Sim already patched')
import importlib.metadata as m
print('scalesim', m.version('scalesim'), m.distribution('scalesim').read_text('direct_url.json'))
PY
echo; echo "ready.  Every new Terminal:  cd scalesim-only && source .venv/bin/activate"
