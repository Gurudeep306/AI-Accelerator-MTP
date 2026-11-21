#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the whole study and write out/simulation_output.txt.

    python3 run_all.py              everything, about 25 minutes
    python3 run_all.py --quick      skips the SCALE-Sim check, about 3 minutes
    python3 run_all.py --check      only checks the environment, runs nothing

The transcript is printed to the screen and written to out/simulation_output.txt
at the same time, so what is demonstrated on screen and what ends up in the file
are the same thing.
"""
import hashlib
import io
import os
import pathlib
import platform
import socket
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def versions():
    out = {}
    try:
        import torch
        out['torch'] = torch.__version__
    except Exception as e:
        out['torch'] = f'NOT INSTALLED ({e.__class__.__name__})'
    try:
        import numpy
        out['numpy'] = numpy.__version__
    except Exception:
        out['numpy'] = 'NOT INSTALLED'
    try:
        import scalesim
        try:
            from importlib.metadata import version, distribution
            v = version('scalesim')
            src = ''
            try:
                import json as _j
                du = distribution('scalesim').read_text('direct_url.json')
                if du:
                    d = _j.loads(du)
                    ci = d.get('vcs_info', {}).get('commit_id', '')
                    if ci:
                        src = f' from {d["url"]} @ {ci[:10]}'
                    elif d.get('url'):
                        src = f' from {d["url"]}'
            except Exception:
                pass
            out['scalesim'] = v + (src or ' from PyPI')
        except Exception:
            out['scalesim'] = 'installed, version unknown'
    except Exception:
        out['scalesim'] = 'NOT INSTALLED'
    return out


def env_check():
    v = versions()
    print('environment')
    print(f'  host                   : {socket.gethostname()}')
    print(f'  operating system       : {platform.platform()}')
    print(f'  machine                : {platform.machine()}')
    print(f'  python                 : {sys.version.split()[0]} at {sys.executable}')
    for k in ('numpy', 'torch', 'scalesim'):
        print(f'  {k:22} : {v[k]}')
    print(f'  working directory      : ./{ROOT.name}')
    missing = [k for k in ('torch', 'numpy') if 'NOT INSTALLED' in v[k]]
    return v, missing


def input_manifest():
    print('\ninput files, with a checksum of each')
    rows = []
    for p in sorted((ROOT / 'inputs').glob('*')) + sorted((ROOT / 'src' / 'mdl').glob('*.py')):
        if p.name == '__init__.py':
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        rel = p.relative_to(ROOT)
        print(f'  {str(rel):44} {p.stat().st_size:9,} bytes   sha256 {h}')
        rows.append((str(rel), h))
    return rows


def main():
    quick = '--quick' in sys.argv
    only_check = '--check' in sys.argv
    # Re-run the trace and the tables against whatever SCALE-Sim results are
    # already cached, without sweeping any new configurations.
    keep_val = '--keep-sim' in sys.argv or '--keep-validation' in sys.argv

    out_dir = ROOT / 'out'
    out_dir.mkdir(exist_ok=True)
    log = (out_dir / 'simulation_output.txt').open('w', encoding='utf-8')
    real = sys.stdout
    sys.stdout = Tee(real, log)

    t0 = time.time()
    print('=' * 78)
    print('PRE-SILICON LATENCY AND FEASIBILITY STUDY OF AN ANYTIME OBJECT DETECTOR')
    print('=' * 78)
    print(f'run started            : {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'mode                   : {"quick, step 5 skipped" if quick else "full"}')
    print()

    v, missing = env_check()
    if missing:
        print(f'\nSTOP. Missing: {", ".join(missing)}')
        print('Install with:  pip install -r requirements.txt')
        sys.stdout = real
        return 1
    if keep_val:
        quick = True
    if not quick and 'NOT INSTALLED' in v['scalesim']:
        print('\nNOTE: scalesim is not installed, so step 5 cannot run.')
        print('Either install it or use --quick. Continuing in quick mode.')
        quick = True

    input_manifest()

    if only_check:
        print('\nEnvironment looks usable. Run without --check to do the study.')
        sys.stdout = real
        return 0

    import step1_trace
    import step2_crosscheck
    import step3_memory
    import step4_resolutions

    step1_trace.main()
    step2_crosscheck.main()
    step3_memory.main()
    step4_resolutions.main()

    if keep_val:
        print('\n' + '=' * 78)
        print('STEP 5   USING THE SCALE-SIM RESULTS ALREADY CACHED (--keep-sim)')
        print('=' * 78)
        print('No new configuration is simulated. The tables below cover whatever')
        print('is in out/shape_cache/, and say "not simulated" for the rest.')
    elif quick:
        print('\n' + '=' * 78)
        print('STEP 5   SKIPPED (--quick)')
        print('=' * 78)
        print('Only configurations already in out/shape_cache/ will appear below.')
    else:
        print('\n' + '=' * 78)
        print('STEP 5   SCALE-SIM OVER EVERY LAYER, EVERY CONFIGURATION')
        print('=' * 78, flush=True)
        import subprocess
        sp = subprocess.Popen([sys.executable, str(ROOT / 'src' / 'sim_sweep.py'),
                               '--tier', '1'],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1)
        for line in sp.stdout:            # through the Tee, so it reaches the log
            print(line.rstrip())
        sp.wait()

    import step6_results
    step6_results.main()

    import audit
    audit.main()

    print('\n' + '=' * 78)
    print(f'run finished in {time.time()-t0:.1f} seconds')
    print(f'transcript   : out/simulation_output.txt')
    print(f'machine data : out/results.json, out/trace.json, out/resolutions.json,')
    print(f'               out/memory.json, out/crosscheck.json')
    print(f'SCALE-Sim    : out/shape_cache/ '
          f'({len(list((out_dir / "shape_cache").glob("*.json"))) if (out_dir / "shape_cache").exists() else 0} '
          f'simulated layer shapes)')
    print('=' * 78)

    sys.stdout = real
    log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
