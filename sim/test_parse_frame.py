import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gait_gen import parse_frame, make_frame

OK_FIELDS = [41, 152300, 2048, 1900, 300, 150, 2200, 3000]

CASES = [
    (make_frame(41, 152300, [2048,1900,300,150,2200,3000]), ("ok", OK_FIELDS)),
    ("XXX,41,152300,2048,1900,300,150,2200,3000,147",       ("malformed", None)),
    ("INS,41,152300,2048,1900,300,150,2200,3000",           ("malformed", None)),
    ("INS,41,152300,2048,1900,300,150,2200,3000,147,99",    ("malformed", None)),
    ("INS,41,152300,ABC,1900,300,150,2200,3000,147",        ("malformed", None)),
    ("INS,41,152300,2048,1900,300,150,2200,3000,XYZ",       ("malformed", None)),
    ("INS,41,152300,2048,1900,300,150,2200,3000,",          ("malformed", None)),
    ("INS,41,152300,2049,1900,300,150,2200,3000,147",       ("bad_checksum", None)),
    ("INS,41,152300,2048,1900,300,150,2200,3000,14",        ("bad_checksum", None)),
]

passed = failed = 0

for line, expected in CASES:
    result = parse_frame(line)
    if result == expected:
        passed += 1
        print(f"PASS  {line}")
    else:
        failed += 1
        print(f"FAIL  got={result}  want={expected}  line={line}")

print(f"{passed} passed, {failed} failed")
if failed:
    sys.exit(1)