# Insole

Instrumented insole for gait analysis. Six force-sensitive resistors sampled at
100 Hz by an ESP32-S3, streamed over USB serial as ASCII frames, logged to CSV
on the host, and reduced to per-stance features for downstream ML work.

The wire format between board and host is specified in
[framespec.md](framespec.md). That document is the contract; the firmware, the
host logger, and the simulator all implement it and must not diverge from it.

**Status:** board assembled, first real dataset captured. The firmware, frame
codec, host logger, calibration, stance detector, feature extraction,
classifier bake-off and a streaming inference script are written and exercised
end to end against both the simulator and the four real captures in
[data/real/](data/real/). Sensor coordinates are measured (±15 mm).
`MAX_DURATION` is set from real data; the other four detector thresholds are
still simulator-derived `RETUNE` values (see [Open items](#open-items)).

## Repository layout

| Path | Purpose |
| --- | --- |
| [framespec.md](framespec.md) | Wire-format specification. Frame layout, checksum, sample rate, validation gates, design rationale. |
| [firmware/insole/](firmware/insole/) | Arduino sketch for the ESP32-S3. Samples six channels, frames them, paces the sample clock against an absolute deadline. |
| [read_serial.py](read_serial.py) | Host logger. One seam, three line sources (`--source file|serial|ble`): validates frames, writes a CSV, exits nonzero on corruption, loss, or a silent board. |
| [infer_live.py](infer_live.py) | Streaming inference over the same seam: detector → features → persisted classifier, one frame at a time, with live counters for discarded stances, extrapolating frames and s4 zeros. |
| [calibration.py](calibration.py) | Conductance transform, per-sensor force fit, and the relative gain match (`apply_gain_match`, conductance space). Stdlib only. |
| [features.py](features.py) | Per-stance features and centre of pressure, lifted byte-for-byte from the notebook so scripts and tests share them. |
| [discriminant.py](discriminant.py) | LDA/QDA from scratch, Wilson accuracy interval, JSON save/load of a fitted model. |
| [fit_model.py](fit_model.py) | Fits the deployment classifier from `features_sessions.csv` and writes `model_lda.json` with its fit metadata. |
| [bakeoff.py](bakeoff.py) / [bakeoff.md](bakeoff.md) | Session-disjoint LDA / QDA / logistic-regression comparison on the simulated sessions. |
| [analyze_real.py](analyze_real.py), [sim_vs_real.py](sim_vs_real.py) | The real `_02` captures through the pipeline, and the simulator against them. Every number in [docs/sim_vs_real.md](docs/sim_vs_real.md) is printed by one of these. |
| [data/real/](data/real/) | First measured-position gait dataset: stand / walk / fast / shuffle, 60 s each. Read its README before using `_01`. |
| [cal_data/](cal_data/), [gain_match.json](gain_match.json) | Bench calibration captures and the single-point relative gain match derived from them. |
| [docs/](docs/) | Write-ups: calibration notes, sim vs real. |
| [gait_gen.py](gait_gen.py) | Frame codec (`make_frame`, `parse_frame`, `checksum`) plus a synthetic gait generator for hardware-free work. Also supplies `true_stances`, the ground truth the detector is scored against. |
| [make_sessions.py](make_sessions.py) | Multi-session dataset generator. One file per (class, session), seeded per class, for session-disjoint train/test splits. |
| [detector.py](detector.py) | Stance detection: `find_stances`, `merge_close`, `stance_report`, plus the `SENSOR_COLS` / `SENSOR_COORDS` definitions and the tuning constants. Single source for all of these. |
| [heatmap.py](heatmap.py) | Foot-pressure field rendering and per-stance animation. |
| [insole.ipynb](insole.ipynb) | Analysis pipeline: load, clean, window, stance detection, feature extraction, centre of pressure, plots. |
| `test_*.py`, [sim/test_parse_frame.py](sim/test_parse_frame.py) | Tests. Each runs directly (`python test_x.py`, PASS/FAIL lines, nonzero exit on failure) and under `python -m pytest`. See [Tests](#tests). |
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

Python 3.10 or newer. From the repository root:

```
pip install -e .                  # numpy, pandas; makes the modules importable anywhere
pip install -e .[hw]              # + pyserial, bleak  (live capture)
pip install -e .[analysis]        # + matplotlib, scipy, scikit-learn
pip install -e .[test]            # + pytest
```

[pyproject.toml](pyproject.toml) is the smallest thing that makes
`import detector` work from any working directory: an editable install of the
flat-layout modules, no `src/` or package directory. Without it, every script
and test still works when run from the repository root.

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

Writes five 60-second captures at 100 Hz:

| File | Stride period | Purpose |
| --- | --- | --- |
| `sim_walk.txt` | 1.0 s | Baseline walking cadence. |
| `sim_fast.txt` | 0.6 s | Faster cadence, for class separation in the feature plots. |
| `sim_shuffle.txt` | 0.5 s | Short, lightly loaded strides. Catches a `MIN_DURATION` set high enough to annihilate brief stances. |
| `sim_dropout.txt` | 1.0 s | Walk with channel 0 dead from 20 s to 40 s. A disconnected FSR reads flat zero, not noisy zero. |
| `sim_stand.txt` | n/a | Static load, no steps. Truth is `[]`, so any detection is a false positive. |

The last three are adversarial on purpose: each one fails loudly for a
different mis-set detector constant, which a walk-only fixture would not catch.

Cadence is a parameter, not an edited constant. `cycle_s` defaults to `None`
and is resolved by `resolve_cycle(mode, cycle_s)`: an explicit value always
wins, otherwise the mode picks the default (0.5 s for `shuffle`, `CYCLE_S`
otherwise). `sensor_value` and `true_stances` resolve on entry and `gait_lines`
forwards untouched, so one call site controls cadence and passing an explicit
`cycle_s` is never silently overridden.

Sensor noise is drawn from an unseeded `random.gauss`, so repeated runs of
`gait_gen.py` produce statistically equivalent but byte-different files.
`make_sessions.py` is the exception: it seeds per class, so its sessions
rebuild byte-identical and are therefore gitignored rather than committed.

To verify the parser against every rejection path in the spec, and the detector
against the five streams:

```
python sim/test_parse_frame.py
python test_stances.py
```

Each prints `PASS` or `FAIL` per case and exits non-zero on any failure. The
full set is listed under [Tests](#tests).

## Capture from hardware

[read_serial.py](read_serial.py) takes its transport and paths on the command
line. The constants at the top of the file (`PORT = "COM7"`, `BAUD = 921600`,
`DURATION_S = 60`, `BLE_NAME = "INSOLE"`) are defaults, and the two that vary
between machines have flags:

```
python read_serial.py --source serial --port COM13 --duration 60 out.csv   # tethered USB
python read_serial.py --source ble --duration 60 out.csv                   # Nordic UART over BLE
python read_serial.py sim_walk.txt sim_walk.csv                            # replay a frame log
```

`--source` defaults to `file` when an input path is given and to `ble`
otherwise. Output has the header `seq,ts_us,s0,s1,s2,s3,s4,s5`. At exit it
prints one summary line:

```
source=serial valid=6000 malformed=0 empty=0 bad_checksum=0 seq_breaks=0 lost=0 loss=0.00% timing_breaks=0 status=0 source_drops=0 capture_s=60.0 device_s=60.0
```

and exits non-zero on any corrupted frame (every transport), on any lost frame
over USB or file, on more than 2% loss over BLE, or when a **live source goes
silent for 3 s** (`STALL_S`): a board that stops sending while the link stays
up used to produce a clean, short file and exit 0, which is the mode that
ruins a dataset. That watchdog prints `FAIL: stalled -- ...` and exits 1.

The three sources sit behind one seam, `make_source()`; everything after it
consumes an iterator of strings. `infer_live.py` imports the same seam, so
switching it between a file, USB and BLE is an argument, not an edit.

## Streaming inference

[infer_live.py](infer_live.py) runs the detector, the feature extractor and the
persisted classifier on frames as they arrive:

```
python infer_live.py sim_walk.txt --label walk                # replay a frame log
python infer_live.py data/real/walk02.csv --label walk        # replay a CSV capture
python infer_live.py --source serial --port COM13 --duration 60
python infer_live.py --source ble --duration 60 --out preds.csv
```

It prints one line per completed stance (span, features, predicted class) and,
every 5 s regardless, the running counters: valid / bad frames, stances
completed, **stances discarded for exceeding `MAX_DURATION`**, frames where the
gain match is extrapolating (any sensor above `calibration.CAL_MAX_COUNTS`),
and frames where s4 reads 0 (below its activation threshold, not missing). A
per-stage `perf_counter` table is printed at exit. The exit code is
`read_serial`'s, plus 1 on a stall.

`test_infer_live.py` asserts that the streaming path produces features
bit-identical to `features.extract_features` on the CSV `read_serial.py`
writes from the same bytes, on every capture in the repository.

The classifier (`model_lda.json`, from [fit_model.py](fit_model.py)) is fitted
on **simulated** sessions. On real captures the recipe scores far below the
majority floor, so its predictions on real gait are a plumbing check until a
real training set exists.

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

## Fault handling

`gait_gen.py` can inject three link faults into any simulated stream, so the
logger and the streamer are exercised against a misbehaving link without a
board. All three are off by default and the generator is byte-identical with
them off (`test_faults.py` pins the twelve session hashes).

```
python gait_gen.py --out faulty.txt --drop-rate 0.01 --corrupt-rate 0.005 --reset-at 30 --fault-seed 1
python read_serial.py faulty.txt faulty.csv      # counters, exit 1
python infer_live.py faulty.txt                  # stances flagged, exit 1
```

The policy, and where each piece lives:

| fault (flag) | what the wire carries | logger (`read_serial.py`) | streamer (`infer_live.py`) | counter | test |
| --- | --- | --- | --- | --- | --- |
| frame lost (`--drop-rate`) | a sequence gap | drop and count; no row written, nothing reconstructed | same counters; a stance whose span contains a gap is flagged with `gap_frames` beside its prediction and in `--out`; features use the frames that arrived | `lost`, `seq_breaks` | `test_faults.py::test_drop_mode` |
| bad checksum (`--corrupt-rate`) | a well-formed frame with the wrong checksum | drop and count; the frame consumed a sequence slot, so its gap is credited to `bad_checksum`, not counted a second time as loss | same; the stance is one frame shorter and flagged | `bad_checksum` | `test_faults.py::test_corrupt_mode` |
| board reset (`--reset-at`) | boot text, then `SEQ` and `ts_us` restart at 0 | count once; re-seed the sequence and timing validators from the first post-reset frame; not also a seq break, loss or timing break | same; the stance in progress is discarded (its end is unobservable), a complete pending stance is released, the running-median dt and the frame buffer are cleared, later stances carry the next `epoch` | `resets` | `test_faults.py::test_reset_mode` |

Both consumers run the same `read_serial.FrameValidator`, and
`test_faults.py::test_logger_streamer_consistency` feeds one stream carrying
all three faults to both, over a file and over the fake serial source, and
asserts identical counters. The accounting identity it checks is
`valid + lost + bad_checksum == frames the board emitted`, up to frames lost
right at a reset boundary, which no host can see. Sensor values are never
imputed, which is the same rule as for s4's below-threshold zeros.

Exit code under injected faults (unchanged semantics plus one addition):
`malformed`, `bad_checksum` or `empty` fail on every transport; `lost`,
`seq_breaks` or `timing_breaks` fail on file and serial, while BLE tolerates
up to 2 % loss; `resets` now fails on every transport, because a reboot
mid-capture leaves a file whose time axis restarts, and over native USB CDC
or BLE the boot text never reaches the host, so the clock is the only
evidence.

What real hardware can still do that this does not anticipate: well-formed
frames carrying wrong physics (an intermittent channel as in the `_01`
captures, the single-frame correlated dips, FSR relaxation drift); a BLE link
left half-open, where notifications stop without a disconnect and only the
inactivity watchdog sees the symptom; and a brownout on battery, which is a
reset the host may never get a frame from.

## Analysis pipeline

[insole.ipynb](insole.ipynb) runs on a saved CSV and is written for Colab.
Hardware work is local, because the board is on a local USB port; analysis of a
saved file can run anywhere.

Stages:

1. **Load and clean.** Sessions load into one frame keyed by source path, with
   `ts_us` as the firmware recorded it (an earlier version overwrote it with
   the 100 Hz sample index, which hid timing jitter). Readings outside the
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
`SENSOR_COORDS`. Those are now measured positions, not placeholders:
`detector.py` holds the raw millimetre table `SENSOR_MM`, taken on a right
insole lying top-up with y from the heel end and x from the medial edge, and
derives `SENSOR_COORDS` from it.

The insole measures **274 x 91 mm (10.8 x 3.6 in)**. Both axes are normalised
by the *length*, so a coordinate times 274 gives millimetres back and the
insole occupies `x` in `[0, 0.332]`, `y` in `[0, 1]`. Normalising each axis by
its own extent would stretch the width by 3.01x and corrupt every distance,
angle and CoP path length; `heatmap.py` rescales `FOOT_OUTLINE` onto the same
box so the two cannot drift apart.

Two independent measurement passes over the same six sensors disagreed by
12-22 mm, so the positions carry roughly **+/-15 mm**. Every CoP number scales
with them and inherits that uncertainty.

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

Excluded from version control by [.gitignore](.gitignore): `data/`, all `*.csv`,
and the generated `sim_<label>_<NN>.txt` sessions from `make_sessions.py`.

The five unnumbered `sim_*.txt` from `gait_gen.py` are committed. They are
small, generated from code in this repository, and load-bearing in two places:
`test_stances.py` scores against them, and the notebook's Colab bootstrap
rebuilds every `sim_*.csv` from them after a fresh clone. Ignoring them would
leave a clone with nothing to rebuild from.

Nothing in the pipeline reads `readings.csv` any more; that name is reserved
for raw hardware capture from [read_serial.py](read_serial.py). The notebook
reads the `sim_*.csv` its bootstrap regenerates and writes features to
`features_walk.csv`.

## Tests

```
python -m pytest -q          # everything, from the repository root
python test_stances.py       # or any one script, for its PASS/FAIL lines
```

| Script | Guards |
| --- | --- |
| [test_stances.py](test_stances.py) | Detector counts against `gait_gen.true_stances` on the five sim streams; `merge_close`. |
| [test_geometry.py](test_geometry.py) | `SENSOR_COORDS` derived from `SENSOR_MM`; closed-form CoP answers; the documented all-zero `(nan, nan)`. |
| [test_calibration.py](test_calibration.py) | Force fit recovery, flags, saturation, and that the gain match multiplies conductance, not counts. |
| [test_capture_window.py](test_capture_window.py) | The capture window is anchored on the first line, not process start. |
| [test_discriminant.py](test_discriminant.py) | LDA/QDA against sklearn; Wilson interval; weighted pooled covariance; the `n_k <= p` guard. |
| [test_infer_live.py](test_infer_live.py) | Streaming == batch features on every capture; read boundaries; malformed frames; all-zero frames; `MAX_DURATION` discards; file/serial/BLE parity; no state leaks; the stall watchdog. |
| [sim/test_parse_frame.py](sim/test_parse_frame.py) | The frame codec's rejection paths (framespec.md section 8). |

Simulated data is not evidence about hardware: `gait_gen`'s constants, the
detector thresholds and the tests over them were co-evolved, so any number
measured on sim data is internally consistent by construction.

## Open items

- `SENSOR_COORDS` now comes from measured positions, but the +/-15 mm spread
  between the two measurement passes is large next to a 91 mm width. A third
  pass, or a jig that fixes the sensors to known locations, would tighten it.
- Physical FSR placement still to be confirmed against the channel order in
  framespec.md section 3. Assembly fixed which GPIO carries which channel; it
  did not establish that channel `sN` is the sensor measured at `SENSOR_MM[sN]`,
  and a swapped pair would move the CoP without changing any total force.
- `MAX_DURATION` was re-set to 200 frames from the real `_02` captures.
  `T_ON`, `T_OFF`, `MIN_DURATION` and `GAP_MERGE` are still the
  simulator-swept values and marked `RETUNE`; `T_OFF` is the next candidate
  (shuffle's natural runs are longer than walk's, which reads as adjacent
  contacts merging). Every stance count in `test_stances.py` is a fixture of
  the simulator, not evidence about hardware.
- The classifier is trained on simulated sessions only. Four minutes of real
  data, one trial per class, is not a training set; a real one is the
  prerequisite for any classifier claim.
- The gain match is extrapolating on most loaded real frames: the highest
  calibration sample was 824 counts and walking peaks reach ~1600-2000.
- The simulator's gait model (sine-shaped load over fixed phase windows) is an
  assumption about foot loading, not a measurement. The detector currently only
  proves it works against that assumption.
- framespec.md refers to `host/read_serial.py` and `sim/gait_gen.py`; both now
  live at the repository root.
