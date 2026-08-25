"""
compare_captures.py — prove the BLE capture is byte-equivalent to the serial one.

How to use it (this is the whole point — ONE board, TWO simultaneous readers).
No copying and no editing: read_serial.py takes the source and the output path
on the command line.

  1. Open two terminals, both in this directory, board plugged in over USB.

  2. Terminal A — start this one FIRST, it has to scan and connect:

         python read_serial.py --source ble readings_ble.csv

     Wait for its "BLE: connected to INSOLE after N.Ns discovery" line.

  3. Terminal B — start within a second or two of that line:

         python read_serial.py --source serial readings_serial.csv

  4. Let both run the full DURATION_S (60 s). Each prints its own summary; the
     capture_s figures should be within a second or so of each other and of 60.

  5. python compare_captures.py

The file names above are the ones this script reads. Change them here and in
the two commands together, or not at all.

Because Serial output stays on unconditionally, both files are views of the
SAME frames. Any sensor value that differs for the same seq is a reassembly
bug, full stop — there is no benign explanation.

    pip install pandas
"""

import sys
import pandas as pd

a = pd.read_csv("readings_serial.csv")
b = pd.read_csv("readings_ble.csv")

print(f"serial rows: {len(a)}   ble rows: {len(b)}")

m = a.merge(b, on="seq", suffixes=("_ser", "_ble"))
print(f"overlapping seq values: {len(m)}")

if len(m) == 0:
    print("FAIL: no overlap — the two captures did not run at the same time")
    sys.exit(1)

bad = []
for col in ["ts_us", "s0", "s1", "s2", "s3", "s4", "s5"]:
    diff = m[m[f"{col}_ser"] != m[f"{col}_ble"]]
    if len(diff):
        bad.append((col, len(diff)))

if bad:
    print("FAIL: value mismatches on identical seq numbers:")
    for col, n in bad:
        print(f"   {col}: {n} rows differ")
    print(m.head(10).to_string())
    sys.exit(1)

coverage = 100.0 * len(m) / len(a) if len(a) else 0.0
print(f"PASS: every overlapping frame identical. "
      f"BLE captured {coverage:.2f}% of the frames serial saw.")
