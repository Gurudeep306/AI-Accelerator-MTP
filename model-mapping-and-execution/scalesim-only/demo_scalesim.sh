#!/usr/bin/env bash
# Video demo: SCALE-Sim v3 itself, run live on our 4x4 output-stationary array.
#   cd scalesim-only && source .venv/bin/activate && bash demo_scalesim.sh      (Enter = next scene)
#   bash demo_scalesim.sh 4        # one scene only
cd "$(dirname "$0")"
B=$'\033[1m'; C=$'\033[36m'; Y=$'\033[33m'; D=$'\033[2m'; R=$'\033[0m'
scene() { clear; printf "\n${B}${C}  SCENE %s  ·  %s${R}\n${D}  %s${R}\n\n" "$1" "$2" "$3"; }
next()  { printf "\n${D}  [Enter] next${R}"; read -r _; }
run()   { printf "${Y}\$ %s${R}\n" "$*"; eval "$@"; }
want()  { [ -z "${1:-}" ] || [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
ONLY="${1:-}"

if want 1; then scene 1 "SCALE-Sim v3, installed from its source repository" "pinned to commit 9f98c43, the version the project used"
run "pip show scalesim | head -3"
run "python3 -c \"import importlib.metadata as m; print(m.distribution('scalesim').read_text('direct_url.json'))\""
next; fi

if want 2; then scene 2 "The accelerator, as SCALE-Sim sees it" "accel_4x4_os.cfg: a 4x4 array, output stationary, on-chip buffers, bandwidth mode CALC"
run "sed -n '4,16p;/run_presets/,\$p' accel_4x4_os.cfg"
next; fi

if want 3; then scene 3 "The layers, as SCALE-Sim sees them" "one row per layer. IFMAP = H + (k-1) because SCALE-Sim does not pad; Channels = Cin / 2 because each INT8 PE does 2 per cycle"
run "sed 's/,/  |  /g' demo_4x4_int8.csv"
next; fi

if want 4; then scene 4 "Run SCALE-Sim" "its own command line, nothing of mine in between (about 45 seconds)"
rm -rf out_demo
run "python3 -m scalesim.scale -c accel_4x4_os.cfg -t demo_4x4_int8.csv -l demo_4x4_int8_layout.csv -p out_demo -i conv"
next; fi

if want 5; then scene 5 "What SCALE-Sim wrote" "a report per run, and a cycle-by-cycle trace per layer"
run "ls out_demo/run out_demo/run/layer1"
echo; run "sed 's/, */  |  /g' out_demo/run/COMPUTE_REPORT.csv"
echo; printf "${D}  'Total Cycles' (3rd column) is the compute time we use. One layer by hand:\n"
printf "  022_elan_p5_bb-blocks-0: 20x20 output, 64 filters, K' = 3x3x32 = 288\n"
printf "  ceil(400/4) x ceil(64/4) x (288 + 6) - 1 = 100 x 16 x 294 - 1 = 470,399${R}\n"
echo; run "head -6 out_demo/run/layer1/IFMAP_SRAM_TRACE.csv"
next; fi

if want 6; then scene 6 "Everything exit 1 needs: 48 layers" "run earlier with:  ./run4x4.sh exit1_4x4_int8.csv | tee exit1_int8.log   (about 2 hours)"
if [ -f exit1_int8.log ]; then
  run "tail -12 exit1_int8.log"
else
  echo "  exit1_int8.log not found: start the long run first (see README.md)"
fi
next; fi

if want 7; then scene 7 "The same in INT4" "run earlier with:  ./run4x4.sh exit1_4x4_int4.csv | tee exit1_int4.log"
if [ -f exit1_int4.log ]; then run "tail -12 exit1_int4.log"; else echo "  exit1_int4.log not found (optional run)"; fi
next; fi

if want 8; then scene 8 "Result" ""
cat <<TXT
  SCALE-Sim v3, 4x4 output-stationary array, all layers exit 1 needs:
    INT8: 89,236,904 cycles  ->  446 ms at 200 MHz  (about 2.2 frames per second)
    INT4: 46,515,940 cycles  ->  233 ms at 200 MHz  (about 4.3 frames per second)
  Stall cycles 0 in every run (CALC mode: ideal memory).  Every number equals the cycle formula.
TXT
echo; fi
