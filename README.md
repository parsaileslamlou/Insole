# Insole

Host-side code for reading a smart-insole board over USB serial and logging
its output to CSV, for later ML work.

## Where things run

| Task                              | Where            | Why                                                    |
| --------------------------------- | ---------------- | ------------------------------------------------------ |
| Real acquisition / logging        | **Local (VS Code)** | The board talks over this machine's USB port. Colab runs on a Google server and can't see that port. |
| ML on a saved CSV                 | **Colab**        | Once you have real readings on disk, upload the CSV and crunch it there. |

Rule of thumb: **touching hardware → local; crunching a saved file → Colab.**

## Setup (local)

1. Install VS Code, Python (python.org), and the official Python extension.
2. Open this folder in VS Code.
3. Install the serial library:

   ```bash
   pip install pyserial
   ```

   Note: the package is **pyserial**, but in code you `import serial`.
   Never `pip install serial`.

## Usage

```bash
python read_serial.py --list                 # show available serial ports
python read_serial.py                         # auto-pick a port, log to data/
python read_serial.py --port /dev/tty.usbmodem1101 --baud 115200
python read_serial.py --out data/session1.csv
```

Recorded CSVs land in `data/` (gitignored). Upload one of those to Colab
for the ML stage.
