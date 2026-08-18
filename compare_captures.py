"""
compare_captures.py — prove the BLE capture is byte-equivalent to the serial one.

How to use it (this is the whole point — ONE board, TWO simultaneous readers):

  1. Copy read_serial.py to read_serial_ble.py.
     In the copy set:  SOURCE = "ble"     OUT_CSV = "readings_ble.csv"
     In the original set: SOURCE = "serial"  OUT_CSV = "readings_serial.csv"
  2. Open two terminals. Start the BLE one first (it has to scan and connect),
     then the serial one within a second or two.
  3. Let both run the full DURATION_S.
  4. python compare_captures.py

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
