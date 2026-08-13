# Insole

Instrumented insole for gait analysis. Six force-sensitive resistors sampled at
100 Hz by an ESP32-S3, streamed over USB serial as ASCII frames, logged to CSV
on the host, and reduced to per-stance features for downstream ML work.

The wire format between board and host is specified in
[framespec.md](framespec.md). That document is the contract; the firmware, the
host logger, and the simulator all implement it and must not diverge from it.

**Status:** pre-hardware. The firmware, frame codec, host logger, and analysis
pipeline are written and exercised end to end against a simulator. Sensors are
not yet attached, so the ADC pin mapping and the physical sensor coordinates
are provisional (see [Open items](#open-items)).

## Repository layout

| Path | Purpose |
| --- | --- |
| [framespec.md](framespec.md) | Wire-format specification. Frame layout, checksum, sample rate, validation gates, design rationale. |
| [firmware/insole/](firmware/insole/) | Arduino sketch for the ESP32-S3. Samples six channels, frames them, paces the sample clock against an absolute deadline. |
| [read_serial.py](read_serial.py) | Host logger. Reads frames from the serial port, validates them, writes `readings.csv`. |
| [gait_gen.py](gait_gen.py) | Frame codec (`make_frame`, `parse_frame`, `checksum`) plus a synthetic gait generator for hardware-free work. |
| [heatmap.py](heatmap.py) | Foot-pressure field rendering and per-stance animation. |
| [insole.ipynb](insole.ipynb) | Analysis pipeline: load, clean, window, stance detection, feature extraction, centre of pressure, plots. |
| [sim/test_parse_frame.py](sim/test_parse_frame.py) | Parser test cases covering each rejection path in framespec.md section 8. |
| [scripts/](scripts/) | Bench utilities: noise-baseline capture and per-channel noise statistics. |

## Hardware

ESP32-S3-DevKitC-1 (WROOM-1). Six FSRs on ADC1 channels, read at 12-bit
resolution with 8x oversampling per sample. The QT Py ESP32-S3 was evaluated
and rejected: it breaks out only two ADC1 pins, and ADC2 is unusable while WiFi
is active. Rationale is in the framespec appendix.

Channel order is fixed by [framespec.md](framespec.md) section 3 (heel, lateral
midfoot, 1st/3rd/5th metatarsal heads, hallux) and must not be re-derived
anywhere else in the codebase.

## Frame format

One newline-terminated ASCII line per sample:

```
INS,41,152300,2048,1900,300,150,2200,3000,147
```

Ten fields: sync token, uint16 sequence number, int64 microsecond timestamp,
six raw 12-bit ADC counts, and an additive checksum modulo 256 over the eight
integer fields. Full field table, checksum derivation, and the reasoning behind
each choice are in [framespec.md](framespec.md).

## Requirements

Python 3.10 or newer.

```
pip install pyserial numpy pandas matplotlib
```

`pyserial` is required only for capture from hardware. The package installs as
`pyserial` but imports as `serial`; `pip install serial` is a different,
unrelated package.

Firmware is built with the Arduino IDE or arduino-cli against the ESP32 board
package. The sketch is [firmware/insole/insole.ino](firmware/insole/insole.ino).

## Running without hardware

The simulator produces frames in the same format the firmware emits, so the
parser, the logger, and the whole analysis pipeline can be exercised with no
board attached.

```
python gait_gen.py
```

Writes two 60-second captures at 100 Hz:

| File | Stride period | Purpose |
| --- | --- | --- |
| `sim_walk.txt` | 1.0 s | Baseline walking cadence. |
| `sim_fast.txt` | 0.6 s | Faster cadence, for class separation in the feature plots. |

Cadence is a parameter, not an edited constant: `gait_lines(duration_s,
mode="walk", cycle_s=CYCLE_S)` threads `cycle_s` down to `sensor_value`, so two
captures at different cadences come from the same code path and their
provenance is recoverable from the call site.

Sensor noise is drawn from an unseeded `random.gauss`, so repeated runs produce
statistically equivalent but byte-different files.

To verify the parser against every rejection path in the spec:

```
python sim/test_parse_frame.py
```

Nine cases, run from the repository root. Each prints `PASS` or `FAIL` with the
offending line.

## Capture from hardware

[read_serial.py](read_serial.py) is configured by the constants at the top of
the file, not by command-line arguments:

| Constant | Default | Meaning |
| --- | --- | --- |
| `PORT` | `COM12` | Serial port of the USB-UART bridge. |
| `BAUD` | `115200` | Must match `Serial.begin()` in the firmware. |
| `DURATION_S` | `60` | Capture window in seconds. `None` runs until interrupted. |

```
python read_serial.py
```

Writes `readings.csv` with header `seq,ts_us,s0,s1,s2,s3,s4,s5`, then reports
counters and exits:

```
valid=6000 malformed=0 empty=0 bad_checksum=0 seq_breaks=0 timing_breaks=0
```

Exit status is non-zero if no frames were valid or if any anomaly counter is
non-zero, so a capture can be used as a pass/fail gate in a bring-up script.

`file_lines()` in the same module reads frames from a text file instead of the
port, using the identical validation path. It is the seam for replaying a
simulator capture or a saved raw log; nothing currently selects it, so using it
means editing the `source` assignment.

### Noise baseline

With sensors attached but unloaded:

```
mkdir data
python scripts/capture_noise.py
python scripts/noise_stats.py data/noise_floor_30s.csv
```

`capture_noise.py` writes a 30-second capture to `data/noise_floor_30s.csv` and
deliberately touches neither `read_serial.py` nor `readings.csv`; it carries its
own copy of the parser so a bench measurement cannot be affected by pipeline
edits. `noise_stats.py` reports per-channel mean, standard deviation, min, max,
peak-to-peak, and a count of samples beyond five standard deviations, which is
what sets a defensible `T_ON` / `T_OFF` in the stance detector.

## Analysis pipeline

[insole.ipynb](insole.ipynb) runs on a saved CSV and is written for Colab.
Hardware work is local, because the board is on a local USB port; analysis of a
saved file can run anywhere.

Stages:

1. **Load and clean.** Sessions load into one frame keyed by source path, with
   `ts_us` reconstructed from the 100 Hz sample index. Readings outside the
   valid 0-4095 ADC range are dropped and the count reported.
2. **Window.** Fixed-length overlapping windows over the six channels.
3. **Stance detection.** Hysteresis on total force across the six channels:
   `T_ON` to enter a stance, a lower `T_OFF` to leave it, and a minimum
   duration to reject brief spikes. Two thresholds rather than one prevent
   chatter around a single level.
4. **Scoring.** Detected stances are matched against ground truth by fractional
   overlap, greedily and without reuse, yielding precision and recall.
   `gait_gen.true_stances` supplies ground truth for simulated captures.
5. **Features.** Per stance: peak force, time to peak, contact time, loading
   rate, and impulse, plus centre-of-pressure path length and displacement.
6. **Comparison.** Walk and fast captures are concatenated with a label column
   and their feature distributions plotted per feature.

Centre of pressure is the force-weighted mean of the sensor positions in
`SENSOR_COORDS`, normalised to a unit foot outline. Those coordinates are
provisional and marked `RETUNE`; every CoP number scales with them, so they are
placeholders for measured positions, not results.

### Pressure heatmap

[heatmap.py](heatmap.py) renders a foot-shaped pressure field from the six
readings and can animate a single stance:

```python
import heatmap as H
out, dt = H.animate(cleaned, stances[0], cop_fn=cop_frame)
```

Writes `stance.gif` and returns the frame interval in milliseconds. The field
between sensors is inverse-distance-weighted interpolation across six points,
not measurement; the rendered caption says so, and any reading of the smooth
region between sensors should be treated accordingly.

## Data files

Captures are excluded from version control by [.gitignore](.gitignore): `data/`
and all `*.csv`. Simulator output (`sim_walk.txt`, `sim_fast.txt`) is committed,
because it is generated from code in this repository and is small enough to be
useful as a fixture.

Notebook cells reference `readings.csv` and `readings_fast.csv`, neither of
which is tracked. Regenerate them from the simulator captures, or supply your
own, before running the notebook top to bottom.

## Open items

- ADC1 pin to sensor mapping (six of GPIO1-GPIO10), pending hardware assembly.
  `PINS` in the sketch is provisional and marked `TODO`.
- Physical FSR placement to be confirmed against the channel order in
  framespec.md section 3, and `SENSOR_COORDS` updated to measured positions.
- Stance thresholds `T_ON`, `T_OFF`, and `MIN_DURATION` are tuned against
  simulated data and marked `RETUNE`. They need re-tuning against a real
  capture and a real noise floor.
- `true_stances` still reads the module-level `CYCLE_S` rather than taking a
  `cycle_s` parameter, so ground truth is only correct for default-cadence
  captures.
- framespec.md refers to `host/read_serial.py` and `sim/gait_gen.py`; both now
  live at the repository root.
