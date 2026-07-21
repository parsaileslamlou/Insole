100 Hz. `seq` is uint16 and wraps. `ts_us` is microseconds since capture start.
`s0`–`s5` are raw 12-bit ADC counts (0–4095); calibration is host-side so that
recalibrating never requires a reflash. `cksum` is the sum of the eight field
values mod 256. Full contract in `framespec.md`.

## Where things run

| Task | Where | Why |
| --- | --- | --- |
| Acquisition / logging | Local | The board is on this machine's USB port; Colab can't see it. |
| ML on a saved CSV | Colab | Upload a recorded CSV and work from there. |

Touching hardware → local. Crunching a saved file → Colab.

## Setup

Requires Python 3.9+.

```bash
pip install pyserial
```

The package is **pyserial**; in code you `import serial`. Do not
`pip install serial` — that is a different, unrelated package.

## Running

Configuration is by editing constants at the top of `read_serial.py`; there is
no command-line interface yet.

**Against the simulator (no hardware):**

```bash
python3 gait_gen.py        # writes sim_walk.txt
python3 read_serial.py     # with: source = file_lines("sim_walk.txt")
```

**Against the board:**

1. Find the port: `python3 -m serial.tools.list_ports -v`
2. Set `PORT` and `BAUD` in `read_serial.py` to match the board and
   `Serial.begin()` in the firmware.
3. Set `source = serial_lines(PORT, BAUD)`.
4. Close the Arduino Serial Monitor — one program per port.
5. `python3 read_serial.py`

Both write `readings.csv` with columns `seq,ts_us,s0..s5`, print a counter
summary, and exit nonzero if any fault counter is nonzero or nothing parsed.

## Open items

- Noise floor characterization and `OVERSAMPLE` retune (stage 6).
- Confirm all six channels respond; s5 reads ~90 unloaded, cause unknown.
- Sensor placement (S0=heel … S5=hallux) is provisional until the insole is assembled.
- `gait_gen.py` writes `sim_walk.txt` on import, so every logger run regenerates it.
- Firmware checksum truncates `ts_us` to 32 bits; diverges from the host after
  ~71.5 minutes of continuous capture.