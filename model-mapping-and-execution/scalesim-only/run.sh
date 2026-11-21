#!/usr/bin/env bash
# Run a network through SCALE-Sim and total the result.
#
# SCALE-Sim and bash. Nothing else. No Python beyond the simulator's own
# entry point.
#
#   ./run.sh <config.cfg> <topology.csv> [layers per slice]
#
# Example:
#   ./run.sh accel_12x12.cfg demo_head1_p5.csv        # 7 layers,  ~4 min
#   ./run.sh accel_12x12.cfg answer_head1.csv  8      # 48 layers, ~1 h
#   ./run.sh accel_12x12.cfg full_network.csv  8      # 151 layers
#
# Why slices. SCALE-Sim does not release memory between the layers of one
# call. Over a long network the resident set climbs until the operating system
# kills the process, with no traceback and no reports. Handing it a few layers
# at a time in a fresh process puts a ceiling on that. Set the third argument
# to 0 to run everything in one call and watch it happen.

set -u

commas() {                          # 1847531 -> 1,847,531
    echo "$1" | rev | sed 's/[0-9]\{3\}/&,/g' | rev | sed 's/^,//'
}

CFG=${1:?usage: ./run.sh <config.cfg> <topology.csv> [layers per slice]}
TOPO=${2:?usage: ./run.sh <config.cfg> <topology.csv> [layers per slice]}
SLICE=${3:-8}

BASE=$(basename "$TOPO" .csv)
LAYOUT="${BASE}_layout.csv"
OUT="out_$(basename "$CFG" .cfg)_${BASE}"
TOTAL_LAYERS=$(( $(wc -l < "$TOPO") - 1 ))

[ -f "$LAYOUT" ] || { echo "missing $LAYOUT"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/slices"
echo "network   : $TOPO  ($TOTAL_LAYERS layers)"
echo "machine   : $CFG"
echo "slices    : $SLICE layers each"
echo "results   : $OUT/"
echo

START=$(date +%s)

run_slice() {                       # $1 = first layer, $2 = count, $3 = name
    local first=$1 count=$2 name=$3
    head -1 "$TOPO"   > "$OUT/slices/$name.csv"
    head -1 "$LAYOUT" > "$OUT/slices/${name}_layout.csv"
    tail -n +2 "$TOPO"   | tail -n +"$first" | head -"$count" >> "$OUT/slices/$name.csv"
    tail -n +2 "$LAYOUT" | tail -n +"$first" | head -"$count" >> "$OUT/slices/${name}_layout.csv"

    python3 -m scalesim.scale \
        -c "$CFG" \
        -t "$OUT/slices/$name.csv" \
        -l "$OUT/slices/${name}_layout.csv" \
        -p "$OUT/slices/$name" \
        -i conv -s N > "$OUT/slices/$name.log" 2>&1
    local rc=$?

    # The -s flag is parsed and then ignored by scale.py, so the per-cycle
    # traces are written whatever you pass. They run to hundreds of megabytes.
    find "$OUT/slices/$name" -name '*_TRACE.csv' -delete 2>/dev/null

    local rep="$OUT/slices/$name/run/COMPUTE_REPORT.csv"
    if [ $rc -ne 0 ] || [ ! -f "$rep" ]; then
        echo "  layers $first..$((first+count-1))  FAILED (killed, most likely out of memory)"
        return 1
    fi
    tail -n +2 "$rep" >> "$OUT/COMPUTE_ALL.csv"
    local c
    c=$(tail -n +2 "$rep" | awk -F, '{s+=$3} END {printf "%.0f", s}')
    printf "  layers %3d..%-3d  %16s cycles\n" "$first" "$((first+count-1))" \
           "$(commas "$c")"
    return 0
}

echo "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %," > "$OUT/COMPUTE_ALL.csv"

FAILED=0
if [ "$SLICE" -eq 0 ]; then
    run_slice 1 "$TOTAL_LAYERS" all || FAILED=1
else
    i=1
    while [ "$i" -le "$TOTAL_LAYERS" ]; do
        n=$SLICE
        [ $((i + n - 1)) -gt "$TOTAL_LAYERS" ] && n=$((TOTAL_LAYERS - i + 1))
        run_slice "$i" "$n" "s$(printf '%03d' "$i")" || FAILED=1
        i=$((i + n))
    done
fi

END=$(date +%s)
echo

if [ "$FAILED" -ne 0 ]; then
    echo "At least one slice did not finish. Re-run with a smaller slice size."
    exit 1
fi

CYCLES=$(awk -F, 'NR>1 {s+=$3} END {printf "%.0f", s}' "$OUT/COMPUTE_ALL.csv")
STALL=$(awk -F, 'NR>1 {s+=$4} END {printf "%.0f", s}' "$OUT/COMPUTE_ALL.csv")
NLAY=$(awk -F, 'NR>1 {n++} END {print n}' "$OUT/COMPUTE_ALL.csv")
UTIL=$(awk -F, 'NR>1 {s+=$5; n++} END {printf "%.1f", s/n}' "$OUT/COMPUTE_ALL.csv")

echo "layers simulated   : $NLAY"
echo "total cycles       : $(commas "$CYCLES")"
echo "stall cycles       : $(commas "$STALL")"
echo "mean layer util    : $UTIL %"
echo
echo "time for one frame, before any derating"
for MHZ in 100 150 200 400; do
    awk -v c="$CYCLES" -v m="$MHZ" 'BEGIN {
        ms = 1000 * c / (m * 1e6)
        printf "  at %4d MHz : %9.2f ms   (%7.2f fps)\n", m, ms, 1000 / ms
    }'
done

echo
echo "took $((END - START)) s"
echo "per-layer report : $OUT/COMPUTE_ALL.csv"
