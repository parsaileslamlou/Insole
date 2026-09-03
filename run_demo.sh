#!/usr/bin/env bash
# Simulator-only demonstration of the pipeline: generate -> log -> segment,
# features, predict with the persisted classifier. Deterministic (the stream
# is seeded), leaves nothing behind, prints one DEMO OK line, exits nonzero on
# any failure.
#
#     ./run_demo.sh                       # `python` from the active environment
#     PYTHON=.venv/bin/python ./run_demo.sh
#
# Needs the package installed (pip install -e .) and runs from any directory.
set -euo pipefail

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

"$PYTHON" -c "import insole.infer_live" 2>/dev/null \
    || { echo "DEMO FAIL: insole and its dependencies are not importable by $PYTHON; run: pip install -e ." >&2; exit 1; }

echo "1/3 generate   python -m insole.gait_gen --out walk.txt --noise-seed 1"
"$PYTHON" -m insole.gait_gen --out "$work/walk.txt" --noise-seed 1

echo "2/3 log        python -m insole.read_serial walk.txt walk.csv"
"$PYTHON" -m insole.read_serial "$work/walk.txt" "$work/walk.csv" | tail -1

echo "3/3 predict    python -m insole.infer_live walk.txt --label walk --quiet --out preds.csv"
out=$("$PYTHON" -m insole.infer_live "$work/walk.txt" --label walk --quiet --out "$work/preds.csv")
echo "$out" | grep -E "^(model|features|stances completed|predictions|agreement)"

stances=$(echo "$out" | sed -n 's/^stances completed=\([0-9]*\).*/\1/p')
preds=$(echo "$out" | sed -n 's/^predictions: //p')
rows=$(($(wc -l < "$work/preds.csv") - 1))
if [ -z "$stances" ] || [ "$stances" -le 0 ] || [ "$rows" -ne "$stances" ]; then
    echo "DEMO FAIL: stances=${stances:-none} rows=$rows" >&2
    exit 1
fi
echo "DEMO OK: $stances stances on a 60 s simulated walk, predictions $preds, $rows rows written"
