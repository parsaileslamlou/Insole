"""
read_serial.py — host-side acquisition for the insole board.

Runs LOCALLY (VS Code), because the board talks over this machine's USB port.
Reads lines coming off the board's serial connection, timestamps each one,
prints them, and appends them to a CSV on disk. That CSV is what you later
upload to Colab for the ML stage.

Requires pyserial:   pip install pyserial     (import name is `serial`)

Usage examples:
    python read_serial.py --list                 # show available serial ports
    python read_serial.py                         # auto-pick a port, log to data/
    python read_serial.py --port /dev/tty.usbmodem1101 --baud 115200
    python read_serial.py --out data/session1.csv
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial                      # this is pyserial — NOT `pip install serial`
    from serial.tools import list_ports
except ImportError:
    sys.exit(
        "pyserial is not installed. Run:  pip install pyserial\n"
        "(the package is 'pyserial' but you import it as 'serial')"
    )


def find_ports():
    """Return a list of (device, description) for every serial port seen."""
    return [(p.device, p.description) for p in list_ports.comports()]


def pick_port():
    """Auto-pick a port. If exactly one exists, use it; otherwise ask the user."""
    ports = find_ports()
    if not ports:
        sys.exit("No serial ports found. Is the board plugged in over USB?")
    if len(ports) == 1:
        print(f"Using the only serial port found: {ports[0][0]}")
        return ports[0][0]
    print("Multiple serial ports found:")
    for i, (dev, desc) in enumerate(ports):
        print(f"  [{i}] {dev}  ({desc})")
    choice = input("Pick a port number: ").strip()
    return ports[int(choice)][0]


def main():
    parser = argparse.ArgumentParser(description="Log insole board serial data to CSV.")
    parser.add_argument("--list", action="store_true", help="list serial ports and exit")
    parser.add_argument("--port", help="serial port device (e.g. /dev/tty.usbmodem1101)")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate (default 115200)")
    parser.add_argument("--out", help="output CSV path (default: data/insole_<timestamp>.csv)")
    args = parser.parse_args()

    if args.list:
        ports = find_ports()
        if not ports:
            print("No serial ports found.")
        for dev, desc in ports:
            print(f"{dev}\t{desc}")
        return

    port = args.port or pick_port()

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("data") / f"insole_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening {port} @ {args.baud} baud ...")
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as exc:
        sys.exit(f"Could not open {port}: {exc}")

    print(f"Logging to {out_path}  (Ctrl-C to stop)")
    rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "raw_line"])   # header
        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue                          # timed out with no data
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                ts = datetime.now().isoformat(timespec="milliseconds")
                writer.writerow([ts, line])
                f.flush()                             # don't lose data if it crashes
                rows += 1
                print(f"{ts}  {line}")
        except KeyboardInterrupt:
            print(f"\nStopped. Wrote {rows} rows to {out_path}")
        finally:
            ser.close()


if __name__ == "__main__":
    main()
