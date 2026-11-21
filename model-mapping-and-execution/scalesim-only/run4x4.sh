#!/usr/bin/env bash
# Run the detector on our 4x4 output-stationary array with SCALE-Sim v3's own command line,
# show SCALE-Sim's output live, and add up the result.
#
#   ./run4x4.sh <topology.csv> [layers per call]
#
#   ./run4x4.sh demo_4x4_int8.csv          # 6 real layers, about 1 minute   (live demo)
#   ./run4x4.sh exit1_4x4_int8.csv         # everything exit 1 needs, INT8   (long: start it early)
#   ./run4x4.sh exit1_4x4_int4.csv         # the same in INT4
#   ./run4x4.sh full_network_4x4_int8.csv  # all 151 convolutions
#
# The topology files are already adapted to our chip:
#   IFMAP size = H + (k - 1)        SCALE-Sim does not pad
#   Channels   = ceil(Cin / lanes)  2 lanes per PE in INT8, 4 in INT4
#   rows named ..._g2of6            a big layer cut into filter groups so SCALE-Sim does not run out of
#                                   memory; the groups add up exactly, plus 1 cycle per extra group
set -u
commas() { echo "$1" | rev | sed 's/[0-9]\{3\}/&,/g' | rev | sed 's/^,//'; }

CFG=accel_4x4_os.cfg
TOPO=${1:?usage: ./run4x4.sh <topology.csv> [layers per call]}
SLICE=${2:-1}
BASE=$(basename "$TOPO" .csv)
LAYOUT="${BASE}_layout.csv"
OUT="out_4x4_${BASE}"
N=$(( $(wc -l < "$TOPO") - 1 ))
[ -f "$LAYOUT" ] || { echo "missing $LAYOUT"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/slices"
echo "network   : $TOPO  ($N rows)"
echo "machine   : $CFG  (4x4, output stationary)"
echo "simulator : python3 -m scalesim.scale   ($(python3 -c 'import importlib.metadata as m; print(m.version("scalesim"))'))"
echo
echo "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %, Layer name" > "$OUT/COMPUTE_ALL.csv"
START=$(date +%s)
i=1
while [ "$i" -le "$N" ]; do
    n=$SLICE; [ $((i + n - 1)) -gt "$N" ] && n=$((N - i + 1))
    name="s$(printf '%03d' "$i")"
    head -1 "$TOPO" > "$OUT/slices/$name.csv";   tail -n +2 "$TOPO"   | tail -n +"$i" | head -"$n" >> "$OUT/slices/$name.csv"
    head -1 "$LAYOUT" > "$OUT/slices/${name}_layout.csv"; tail -n +2 "$LAYOUT" | tail -n +"$i" | head -"$n" >> "$OUT/slices/${name}_layout.csv"
    echo "──────── rows $i..$((i + n - 1)) of $N : $(tail -n +2 "$OUT/slices/$name.csv" | cut -d, -f1 | paste -sd ' ' -)"
    python3 -m scalesim.scale -c "$CFG" -t "$OUT/slices/$name.csv" -l "$OUT/slices/${name}_layout.csv" \
            -p "$OUT/slices/$name" -i conv 2>&1 | tee "$OUT/slices/$name.log"
    rep="$OUT/slices/$name/run/COMPUTE_REPORT.csv"
    find "$OUT/slices/$name" -name '*_TRACE.csv' -delete 2>/dev/null    # traces are huge; we only need the report
    [ -f "$rep" ] || { echo "rows $i..$((i + n - 1)) FAILED (most likely out of memory)"; exit 1; }
    paste -d, <(tail -n +2 "$rep" | sed 's/, *$//') <(tail -n +2 "$OUT/slices/$name.csv" | cut -d, -f1) >> "$OUT/COMPUTE_ALL.csv"
    i=$((i + n))
done
END=$(date +%s)

SUM=$(awk -F, 'NR>1 {s+=$3} END {printf "%.0f", s}' "$OUT/COMPUTE_ALL.csv")
EXTRA=$(awk -F, 'NR>1 && $8 ~ /_g[0-9]+of/ && $8 !~ /_g1of/ {n++} END {print n+0}' "$OUT/COMPUTE_ALL.csv")
CYC=$((SUM + EXTRA))
echo
echo "════════════════════════════════════════════════════"
echo "rows simulated          : $N"
echo "sum of Total Cycles     : $(commas "$SUM")"
echo "+ 1 per extra group     : $EXTRA"
echo "cycles for this network : $(commas "$CYC")"
echo "stall cycles            : $(commas "$(awk -F, 'NR>1 {s+=$4} END {printf "%.0f", s}' "$OUT/COMPUTE_ALL.csv")")"
for MHZ in 100 150 200; do
    awk -v c="$CYC" -v m="$MHZ" 'BEGIN { ms = 1000 * c / (m * 1e6); printf "  at %3d MHz : %9.1f ms   (%5.2f fps)\n", m, ms, 1000 / ms }'
done
echo "took $((END - START)) s     per-layer report: $OUT/COMPUTE_ALL.csv"
