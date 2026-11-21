#!/usr/bin/env bash
# Scene-by-scene demo of the model-mapping work, for screen recording.
#   cd model-mapping-and-execution && source .venv/bin/activate && bash demo/demo.sh
# Press Enter to move to the next scene. Run one scene only: bash demo/demo.sh 5
set -e
cd "$(dirname "$0")/.."
B=$'\033[1m'; C=$'\033[36m'; Y=$'\033[33m'; D=$'\033[2m'; R=$'\033[0m'
scene() { clear; printf "\n${B}${C}  SCENE %s  ·  %s${R}\n${D}  %s${R}\n\n" "$1" "$2" "$3"; }
wait_key() { printf "\n${D}  [Enter] next scene${R}"; read -r _; }
run() { printf "${Y}\$ %s${R}\n" "$*"; eval "$@"; }
ONLY="${1:-}"
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

if want 0; then
scene 0 "Mapping our anytime detector onto the 4x4 accelerator" "Paidipati Gurudeep · CS25M111 · model mapping and execution"
cat <<TXT
  What this demo shows, all run live on this laptop:
    1. the detector code I received
    2. running it once and recording all 151 convolutions
    3. checking that against the model author's own layer table
    4. 151 layers -> 37 unique shapes
    5. one tiny layer through SCALE-Sim, with its cycle-by-cycle trace
    6. one real detector layer through SCALE-Sim, checked with a formula
    7. every stored SCALE-Sim result checked against the formula
    8. the latency of all four exits
TXT
wait_key; fi

if want 1; then
scene 1 "The detector code I received" "unchanged copies of the model author's files"
run "ls model-source"
echo; run "grep -n 'def forward_\|self.head[0-9_a-z]* = DetectHead' model-source/gelan_anytime.py"
wait_key; fi

if want 2; then
scene 2 "Run the model once, record every convolution" "a hook on every Conv2d; zeros as input; only shapes matter for cycles"
run "python3 src/step1_trace.py | sed -n '4,28p'"
wait_key; fi

if want 3; then
scene 3 "Check against the model author's layer table" "every layer, every field"
run "python3 src/step2_crosscheck.py | tail -12"
wait_key; fi

if want 4; then
scene 4 "151 layers -> 37 unique shapes" "same shape = same cycles, so each shape is simulated once per format"
run "python3 demo/results.py shapes | head -20"
wait_key; fi

if want 5; then
scene 5 "One tiny layer through SCALE-Sim" "4x4 input, 4 channels, 3x3 kernel, 4 filters, INT8"
run "python3 demo/one_layer.py --toy"
wait_key; fi

if want 6; then
scene 6 "One real detector layer through SCALE-Sim" "20x20x64 input, 3x3 kernel, 64 filters, INT8 (about 15 seconds)"
run "python3 demo/one_layer.py 20 3 1 64 64"
wait_key; fi

if want 7; then
scene 7 "Every stored SCALE-Sim result against the formula" "cycles = ceil(M/4) x ceil(N/4) x (K' + 6) - 1 on our 4x4 array"
run "python3 tools/check_formula.py"
wait_key; fi

if want 8; then
scene 8 "Latency of all four exits" "each exit = sum of the cycles of the layers it runs"
run "python3 demo/results.py exits"
wait_key; fi

if want 9; then
scene 9 "Summary" ""
cat <<TXT
  151 convolutions -> 37 shapes -> 74 SCALE-Sim runs (INT8 + INT4) on a 4x4 output-stationary array
  every result equals the cycle formula exactly
  exit 1 at 200 MHz: 446 ms in INT8, 233 ms in INT4  (about 2 and 4 frames per second)
  next: measure on the board, measure INT4 accuracy, give the table to the controller
TXT
echo; fi
