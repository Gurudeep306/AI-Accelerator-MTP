# -*- coding: utf-8 -*-
"""Where everything lives. Nothing in this package uses an absolute path."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
INPUTS = ROOT / 'inputs'
OUT = ROOT / 'out'
OUT.mkdir(exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load(name):
    return json.loads((INPUTS / name).read_text())


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj))


def read(name):
    return json.loads((OUT / name).read_text())


def rule(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)
