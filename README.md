# Insole

A smart insole for gait analysis: six force-sensitive resistors (Interlink FSR
UX 402) under a right insole, sampled at 100 Hz by an ESP32-S3, streamed to a
laptop over USB serial or BLE as checksummed ASCII frames, logged to CSV,
segmented into stances, reduced to per-stance features, and classified by a
from-scratch LDA/QDA. Every stage has a simulator or a stub behind the same
interface as the hardware, so the whole pipeline runs, and is tested, without
a board. Real captures from the board exist for four activities and are the
first non-circular test of everything the simulator was tuned against.

**Start here:** [notebooks/demo.ipynb](notebooks/demo.ipynb) runs the whole
pipeline without hardware, spec to heatmap, in a few seconds, with the real
captures beside the simulated ones; its outputs are committed so it reads on
GitHub. [docs/writeup.md](docs/writeup.md) is the project in 600 words, and
`./run_demo.sh` runs generate → log → predict on the simulator in one command.

## Data path

```
 6x FSR ──1 kΩ dividers──> ESP32-S3 ADC1 (12-bit, 8x oversample, 100 Hz)
                                  │  INS,SEQ,TS_US,S0..S5,CHECKSUM\n
                     ┌────────────┴────────────┐
                USB CDC serial            BLE Nordic UART
                     └────────────┬────────────┘
                                  │  read_serial.make_source()  ── the seam ──┐
  insole/gait_gen.py  (sim)  ─────┤                                            │
  sim_*.txt fixtures         ─────┘                                            │
                                  ▼                                            │
                   FrameValidator: parse, checksum, seq/timing/reset counters   │
                     ┌────────────┴────────────┐                               │
          insole/read_serial.py         insole/infer_live.py                   │
          (CSV logger, exit code)       (one frame at a time)                  │
                     │                          │                              │
                     ▼                          ▼                              │
            detector.find_stances       detector.StanceTracker ─ same state machine
                     │                          │
                     ▼                          ▼
            features.* on representation B   features.* per stance ─ bit-identical
                     │                          │
                     ▼                          ▼
            discriminant LDA / QDA      models/model_*.json  ─> prediction per stance

  The detector sees raw counts. The features see representation B,
  conductance x = counts / (4095 − counts) per channel, on every source
  (insole/representations.py, chosen on the real captures at stage 20). The
  gain match (models/gain_match.json) is applied per frame in conductance
  space and drives the extrapolation counter only.
```

## Repository layout

| Path | Purpose |
| --- | --- |
| [insole/](insole/) | The package: everything a test or another module imports. `gait_gen` (frame codec + simulator + fault modes), `read_serial` (logger, the transport seam, the validator), `infer_live` (streaming inference), `detector`, `features`, `representations` (what the features see), `splits`, `discriminant`, `calibration`, `fit_calibration`, `heatmap`, `make_sessions`, `paths`. |
| [scripts/](scripts/) | Programs that are run, never imported: `train_real.py` (the real-data classifier and its analysis), `bakeoff.py`, `fit_model.py`, `analyze_real.py`, `sim_vs_real.py`, `sweep_max_duration.py`, `capture_calibration.py`, `compare_captures.py`, `capture_noise.py`, `noise_stats.py`. |
| [tests/](tests/) | Every test. Each runs directly (PASS/FAIL lines, nonzero exit on failure) and under pytest. |
| [notebooks/](notebooks/) | [demo.ipynb](notebooks/demo.ipynb), the run-all demonstration (start here); [insole.ipynb](notebooks/insole.ipynb), the analysis pipeline (Colab-ready). |
| [docs/](docs/) | [writeup.md](docs/writeup.md) (the project in 600 words), [hardware_notes.md](docs/hardware_notes.md) (board, bench, failure modes, what the live bench must show), [frame_spec.md](docs/frame_spec.md) (the wire contract), [calibration_notes.md](docs/calibration_notes.md), [sim_vs_real.md](docs/sim_vs_real.md), [bakeoff.md](docs/bakeoff.md), [real_results.md](docs/real_results.md). |
| [models/](models/) | `gain_match.json` (single-point relative gain match), `model_lda.json`, `model_qda.json` (sim-trained classifiers), `model_lda_real.json`, `model_qda_real.json` (trained on the real captures). |
| [data/real/](data/real/) | Four real captures, `_02` set (training/evaluation) and `_01` set (failure evidence). Read its README first. |
| [data/sim/](data/sim/) | The five committed simulator fixtures `sim_*.txt`; generated CSVs and sessions land here too and are ignored. |
| [cal_data/](cal_data/) | 42 bench calibration captures and their manifest. |
| [figures/](figures/) | Rendered comparison figures. |
| [firmware/insole/](firmware/insole/) | The Arduino sketch and its header. |
| [run_demo.sh](run_demo.sh) | The simulator demo in one command; prints `DEMO OK` or exits nonzero. |

Two invocation forms, used consistently everywhere: package programs run as
`python -m insole.<module>`, scripts run as `python scripts/<name>.py`. Both
resolve repository paths through `insole/paths.py`, never through the working
directory, so they work from anywhere once the package is installed.

## Quickstart (simulator only, no hardware)

Every command below was executed as written. Python 3.10 or newer.

```
git clone https://github.com/parsaileslamlou/Insole.git && cd Insole
python3 -m venv .venv && source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate
pip install -e ".[analysis,test,notebook]"
```

1. Generate a 60 s simulated walk (seeded, so the numbers below reproduce):

   ```
   python -m insole.gait_gen --out demo_walk.txt --noise-seed 1
   wrote demo_walk.txt: 6000 lines, 6000 frames generated
   ```

2. Log it through the validator to a CSV:

   ```
   python -m insole.read_serial demo_walk.txt demo_walk.csv
   source=file valid=6000 malformed=0 empty=0 bad_checksum=0 seq_breaks=0 lost=0 loss=0.00% timing_breaks=0 resets=0 status=0 source_drops=0 capture_s=0.0 device_s=60.0
   ```

3. Stream it through the detector, the features and the persisted classifier:

   ```
   python -m insole.infer_live demo_walk.txt --label walk --quiet
   ...
   features  : representation B (conductance) on every source; the detector sees raw counts; model fitted on 'conductance'
   ...
   stances completed=60 discarded>MAX_DURATION(200)=0 discarded@reset=0 rejected<MIN_DURATION(15)=0 predicted=60 no_prediction=0 stances_with_gaps=0
   frames extrapolating (any sensor > 824 counts)=3420 (57.0%)  s4=0 frames=2388 (39.8%)  all-zero frames=43
   predictions: fast=6  walk=54
   agreement with --label 'walk': 54/60 = 0.9000  (agreement with a typed label, not an accuracy)
   ```

   The default model is the sim-trained LDA. `--model models/model_qda_real.json`
   loads the one trained on the real captures ([docs/real_results.md](docs/real_results.md)).

4. Build the 12 simulated sessions and the feature frame, and run the bake-off
   (writes `data/sim/features_sessions.csv`; the test suite reads it):

   ```
   python scripts/bakeoff.py
   ```

5. Run the tests:

   ```
   python -m pytest -q                 # 44 passed
   python tests/test_stances.py        # or any one file, for its PASS/FAIL lines (293 across the nine files)
   ```

6. Or do steps 1–3 in one command, deterministic, leaving nothing behind
   (bash; on Windows use Git Bash; `PYTHON=.venv/bin/python` if `python` is
   not the venv's):

   ```
   ./run_demo.sh
   ...
   DEMO OK: 60 stances on a 60 s simulated walk, predictions fast=6  walk=54, 60 rows written
   ```

Both notebooks run from a fresh clone: `jupyter nbconvert --to notebook
--execute notebooks/demo.ipynb` (or `insole.ipynb`) from the repository root,
or open one in Colab (the first cell clones and installs). **Colab caveat:** a
tab opened before a push keeps the old tree when you "Save a copy in GitHub".
Always start from a fresh session.

## Status

| Stage | What | Where | Status |
| --- | --- | --- | --- |
| 1 | Wire-format specification | [docs/frame_spec.md](docs/frame_spec.md) | done |
| 2 | Frame codec + gait simulator | `insole/gait_gen.py` | done |
| 3 | Firmware: six-channel 100 Hz sampler | `firmware/insole/` | done |
| 4 | Host logger, noise-floor utilities | `insole/read_serial.py`, `scripts/capture_noise.py`, `scripts/noise_stats.py` | done |
| 5 | Notebook pipeline: load, clean, window | `notebooks/insole.ipynb` | done |
| 6 | Stance detection | `insole/detector.py` | done |
| 7 | Per-stance features and centre of pressure | `insole/features.py` | done |
| 8 | Pressure heatmap and stance animation | `insole/heatmap.py` | done |
| 9 | Simulator regression fixtures and detector tests | `data/sim/`, `tests/test_stances.py` | done |
| 10 | Multi-session sim data, from-scratch LDA/QDA | `insole/make_sessions.py`, `insole/discriminant.py` | done |
| 11 | Model bake-off | `scripts/bakeoff.py`, [docs/bakeoff.md](docs/bakeoff.md) | done |
| 12 | Calibration: single-point relative gain match | `insole/calibration.py`, `cal_data/`, `models/gain_match.json` | done |
| 13 | Real captures and sim-vs-real analysis | `data/real/`, `scripts/analyze_real.py`, [docs/sim_vs_real.md](docs/sim_vs_real.md) | done; the planned weight-shift activity was not collected |
| 14 | Streaming inference over serial and BLE | `insole/infer_live.py`, firmware BLE | done 2026-09-03; live bench run on the board, four of four captures pass (numbers below) |
| 15 | Fault injection and robustness | `insole/gait_gen.py` fault modes, `tests/test_faults.py` | done |
| 16 | Repository consolidation, this README | package layout | done |
| 17 | Demo notebook | [notebooks/demo.ipynb](notebooks/demo.ipynb), `figures/demo/` | done |
| 18 | Writeup | [docs/writeup.md](docs/writeup.md) | done |
| 19 | Final pass, `run_demo.sh`, hardware notes | [run_demo.sh](run_demo.sh), [docs/hardware_notes.md](docs/hardware_notes.md) | done |
| 20 | Classifier trained on the real captures | `scripts/train_real.py`, [docs/real_results.md](docs/real_results.md) | done; a per-session split is not possible with one session per class, so the headline is time-blocked within session |

## Hardware

Bench detail, failure modes and every measurement that is not in a script
are in [docs/hardware_notes.md](docs/hardware_notes.md).

**Board.** ESP32-S3-DevKitC-1 (WROOM-1), Arduino-ESP32 core. The sketch folder
name equals the sketch name (`firmware/insole/insole.ino`). In the Arduino IDE
set *Tools ▸ USB CDC On Boot: Enabled*; without it the board emits nothing on
the USB port. The QT Py ESP32-S3 was evaluated and rejected: it breaks out only
two ADC1 pins, and ADC2 is unusable while the radio is on
([docs/frame_spec.md](docs/frame_spec.md), appendix).

**Pins and dividers.** `PINS = {4, 5, 6, 7, 8, 3}` carry `s0..s5`, all on ADC1
(`firmware/insole/insole.ino`); ADC2 is unusable once BLE is up, which makes
ADC1-only a requirement, not a preference. GPIO3 is a JTAG strapping pin: never
add a pull-up or pull-down to it. Each FSR sits in a divider against 1 kΩ
(`insole/calibration.py`). 12-bit resolution, 8x oversample-and-average per
sample, 100 Hz on an absolute-deadline scheduler, timestamps from
`esp_timer_get_time()` relative to a latched t0.

**Sensor placement** (`insole/detector.py`, `SENSOR_MM`; measured on a right
insole lying top-up, x from the medial edge, y from the heel end; the insole is
274 × 91 mm). Two measurement passes disagreed by 12–22 mm, so every position
carries about ±15 mm, and every centre-of-pressure number inherits it.

| sensor | x (mm) | y (mm) | anatomy |
| --- | --- | --- | --- |
| s0 | 33.0 | 50.8 | heel, medial of the pair |
| s1 | 50.8 | 50.8 | heel, lateral of the pair |
| s2 | 76.2 | 152.4 | lateral midfoot |
| s3 | 81.3 | 185.4 | 5th metatarsal head |
| s4 | 25.4 | 203.2 | 1st metatarsal head |
| s5 | 25.4 | 254.0 | hallux |

**Capture.** Close the Arduino Serial Monitor first (one program per port).
`--port` overrides the default in `insole/read_serial.py`; nothing
machine-local is committed.

```
python -m insole.read_serial --source serial --port COM13 --duration 60 bench_serial.csv
python -m insole.read_serial --source ble --duration 60 bench_ble.csv
python -m insole.infer_live --source serial --port COM13 --duration 60 --out bench_serial_preds.csv
python -m insole.infer_live --source ble --duration 60 --out bench_ble_preds.csv
```

**Pass criteria for the stage-14 live bench:** each 60 s capture reports about
6000 valid frames with `malformed=0 bad_checksum=0 seq_breaks=0
timing_breaks=0 resets=0` over serial, and over BLE `loss` at or under 2 % with
`timing_breaks=0`; every command exits 0; `infer_live` completes stances on
both transports with identical counters to the logger. A stall (`FAIL:
stalled`) means the board went silent for 3 s while the link stayed up. The
firmware prints `# ble ...` status lines once per second; the host counts them
as `status`, not as corruption.

**Live bench, run 2026-09-03 on the board: four of four captures pass.** Right
insole worn, walking (in place or a small figure-8) for the whole of every run;
ESP32-S3 on COM3. Every number below is read from the named log in
`data/bench/`.

| # | run | log | valid | rate | malformed | bad_checksum | seq_breaks | timing_breaks | resets | lost | exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | logger, serial | `serial_logger.log` | 6000 | 100.00 Hz | 0 | 0 | 0 | 0 | 0 | 0 (0.00 %) | 0 |
| 2 | logger, BLE | `ble_logger.log` | 6003 | 100.05 Hz | 0 | 0 | 0 | 0 | 0 | 0 (0.00 %) | 0 |
| 3 | `infer_live`, serial | `serial_preds.log` | 6001 | 100.02 Hz | 0 | 0 | 0 | 0 | 0 | 0 (0.00 %) | 0 |
| 4 | `infer_live`, BLE | `ble_preds.log` | 6003 | 100.05 Hz | 0 | 0 | 0 | 0 | 0 | 0 (0.00 %) | 0 |

Both streaming runs completed 33 stances with `predicted=33 no_prediction=0
stances_with_gaps=0`; run 3 printed `shuffle=31 walk=2`, run 4 `shuffle=14
walk=19`. The default model is sim-trained, so its labels on real walking are
not evidence of anything and the bench does not grade them. Run 2's sequence
numbers are contiguous across all 6003 frames.

All six channels are live. The press captures reach s0 1789, s1 1769, s2 1786,
s3 1746 (`press.csv`) and s4 1953, s5 1661 (`press_medial.csv`, which presses
the two medial pads in a known order with a gap between them so each is
attributed unambiguously — the first sweep missed s5 by pressing the wrong pad,
which looks identical to a dead channel). Both logger runs show all six active
through the walk, s4 lowest at 26–27 % of frames, as its activation threshold
predicts.

**The BLE runs completed only with the board on battery.** Two BLE attempts
with the board powered from this PC's USB cable stalled identically: connected
in 0.7–1.3 s at MTU 517, then the firmware counters showed `notif=7` (21
frames) and `disc=1` roughly 210 ms after `requestConnParams`, while the host
recorded `valid=1` and `FAIL: stalled` (`ble_logger_attempt1_stalled.log`,
`ble_logger_attempt2_stalled.log`; firmware counters in
`ble_diag_after_stall.log` and `ble_diag_after_stall2.log`). The board was not
the fault: between attempts it logged `valid=601` over serial with every
counter zero, and the `conns`/`disc` counters it prints exist only in firmware
at or after the connection-parameter fix, so the reflash this symptom is
otherwise diagnostic of does not apply. Three things changed before the passing
attempt — Windows Bluetooth toggled off and on, the phone's Bluetooth disabled,
and the board moved to a USB power bank — so the cause is narrowed to the
Windows-side central but **not isolated**.

**Calibration.** `models/gain_match.json` is a *single-point relative gain
match*: the six channels' gains matched to their mean at one ~12 N load after
a ≥ 35 min rest, applied in conductance space `x = counts / (4095 − counts)`.
It is not an absolute force calibration. Derivation and limitations are in
[docs/calibration_notes.md](docs/calibration_notes.md); regenerate with
`python -m insole.fit_calibration` (reads `cal_data/calibration_manifest.csv`,
writes `models/gain_match.json`). The highest count any calibration sample
reached is 824 (`calibration.CAL_MAX_COUNTS`, checked against `cal_data/` by
the tests), and loaded walking frames sit above it 62–67 % of the time
(`scripts/analyze_real.py` C3): reported, never clamped.

## Fault handling

`insole/gait_gen.py` can inject three link faults into any simulated stream, so
the logger and the streamer are exercised against a misbehaving link without a
board. All three are off by default and the generator is byte-identical with
them off (`tests/test_faults.py` pins the twelve session hashes).

```
python -m insole.gait_gen --out faulty.txt --drop-rate 0.01 --corrupt-rate 0.005 --reset-at 30 --fault-seed 1
python -m insole.read_serial faulty.txt faulty.csv      # counters, exit 1
python -m insole.infer_live faulty.txt                  # stances flagged, exit 1
```

| fault (flag) | what the wire carries | logger (`insole/read_serial.py`) | streamer (`insole/infer_live.py`) | counter | test |
| --- | --- | --- | --- | --- | --- |
| frame lost (`--drop-rate`) | a sequence gap | drop and count; no row written, nothing reconstructed | same counters; a stance whose span contains a gap is flagged with `gap_frames` beside its prediction and in `--out`; features use the frames that arrived | `lost`, `seq_breaks` | `tests/test_faults.py::test_drop_mode` |
| bad checksum (`--corrupt-rate`) | a well-formed frame with the wrong checksum | drop and count; the frame consumed a sequence slot, so its gap is credited to `bad_checksum`, not counted a second time as loss | same; the stance is one frame shorter and flagged | `bad_checksum` | `tests/test_faults.py::test_corrupt_mode` |
| board reset (`--reset-at`) | boot text, then `SEQ` and `ts_us` restart at 0 | count once; re-seed the sequence and timing validators from the first post-reset frame; not also a seq break, loss or timing break | same; the stance in progress is discarded (its end is unobservable), a complete pending stance is released, the running-median dt and the frame buffer are cleared, later stances carry the next `epoch` | `resets` | `tests/test_faults.py::test_reset_mode` |

Both consumers run the same `read_serial.FrameValidator`, and
`tests/test_faults.py::test_logger_streamer_consistency` feeds one stream
carrying all three faults to both, over a file and over the fake serial
source, and asserts identical counters. The accounting identity it checks is
`valid + lost + bad_checksum == frames the board emitted`, up to frames lost
right at a reset boundary, which no host can see. Sensor values are never
imputed, which is the same rule as for s4's below-threshold zeros.

Exit code under injected faults: `malformed`, `bad_checksum` or `empty` fail
on every transport; `lost`, `seq_breaks` or `timing_breaks` fail on file and
serial, while BLE tolerates up to 2 % loss; `resets` fails on every transport,
because a reboot mid-capture leaves a file whose time axis restarts, and over
native USB CDC or BLE the boot text never reaches the host, so the clock is
the only evidence.

What real hardware can still do that this does not anticipate: well-formed
frames carrying wrong physics (an intermittent channel as in the `_01`
captures, the single-frame correlated dips, FSR relaxation drift); a BLE link
left half-open, where notifications stop without a disconnect and only the
inactivity watchdog sees the symptom; and a brownout on battery, which is a
reset the host may never get a frame from.

## Results

Every number here is printed by the named script; regenerate before quoting.

**Simulated bake-off** (`python scripts/bakeoff.py`; [docs/bakeoff.md](docs/bakeoff.md)).
Two CoP features only, features under representation B, session-disjoint
split, 270 held-out stances, majority floor 0.4296. Logistic regression
0.9185, LDA 0.9185, QDA 0.9296, Wilson 95 % intervals in the doc. The three
models' fast→walk errors are 7 / 5 / 5 rows with 4 in common. Simulated data is
not evidence about hardware: the generator's constants, the detector
thresholds and the tests over them were co-evolved, so this result is
internally consistent by construction.

**Sim-trained model on the real captures** (`python scripts/sim_vs_real.py`,
D2). On the 113 real moving stances (walk 35, fast 48, shuffle 30) the
sim-trained LDA scores 0.3097 and QDA 0.2566, below the 0.4248 majority floor.
That is the expected outcome for a model fitted on a simulator, and it is a
plumbing check, not a classifier result.

**Classifier trained on the real captures** (`python scripts/train_real.py`;
[docs/real_results.md](docs/real_results.md), every number regenerated by
the script). Headline, chosen by a rule fixed before any result was seen
(CoP-only features; the representation and model with the best
contiguous-block CV accuracy): representation B, QDA, time-blocked test
accuracy 0.6444 [0.4984, 0.7678] (29/45) against a test floor of 0.4222,
block-CV 0.6372, random stance-level split 0.6433. This is a within-session
number: one session per class, the first 60 % of each session's stances
train and the last 40 % test, so it carries the leakage that implies and is
not a per-session result. With all seven features (contact time and its
relatives included) the best cell reaches 0.9111 [0.7927, 0.9649], riding on
cadence. The doc lists every misclassified stance with a figure, measures the
s4-zero and gain-match effects on the CoP in millimetres, and checks peak
force against onset time.

**Real captures through the pipeline** (`python scripts/analyze_real.py`;
[docs/sim_vs_real.md](docs/sim_vs_real.md)). Stances at the committed
thresholds: stand 0, walk 35, fast 48, shuffle 30 (`tests/test_stances.py`
pins them). s4 reads 0 on 52–56 % of moving frames because it has the highest
activation threshold of the six; on those frames the CoP is a five-sensor
centroid biased laterally by 33.6–36.7 mm against a 91 mm width. The stance
detector's `MAX_DURATION` was raised from 120 to 200 on this data
(`python scripts/sweep_max_duration.py`): at 120 it discarded 17 of 35 walk and
28 of 30 shuffle contacts outright, because real contacts here run 84–164
frames while no simulated stance exceeds 60.

## Analysis notebook

[notebooks/insole.ipynb](notebooks/insole.ipynb) runs on the saved CSVs and is
written for Colab; its first cell clones the repository, installs the package,
and rebuilds the simulated CSVs from the committed streams. Locally, execute it
from the repository root. Stages: load and clean (with `ts_us` exactly as the
firmware recorded it; an earlier version overwrote it with the sample index,
which hid timing jitter and was caught only when real data arrived), window,
stance detection, scoring against `gait_gen.true_stances`, features, and the
walk-versus-fast feature plots. The pressure heatmap between the six sensors
is inverse-distance-weighted interpolation, not measurement, and the render
says so.

## Tests

```
python -m pytest -q                 # everything, from the repository root (44 functions)
python tests/test_faults.py         # or any one file, for its PASS/FAIL lines
```

| File | Guards | PASS lines |
| --- | --- | --- |
| [tests/test_stances.py](tests/test_stances.py) | Detector counts against `gait_gen.true_stances` on the five sim streams; `merge_close`; the real `_02` stance counts 0 / 35 / 48 / 30. | 19 |
| [tests/test_geometry.py](tests/test_geometry.py) | `SENSOR_COORDS` derived from `SENSOR_MM`; closed-form CoP answers; the documented all-zero `(nan, nan)`. | 51 |
| [tests/test_calibration.py](tests/test_calibration.py) | Force fit recovery, flags, saturation; the gain match multiplies conductance, not counts; the six shipped corrections. | 55 |
| [tests/test_capture_window.py](tests/test_capture_window.py) | The capture window is anchored on the first line, not process start. | 13 |
| [tests/test_discriminant.py](tests/test_discriminant.py) | LDA/QDA against sklearn; Wilson interval; weighted pooled covariance; the `n_k <= p` guard. | 7 |
| [tests/test_infer_live.py](tests/test_infer_live.py) | Streaming == batch features on every capture; read boundaries; malformed frames; all-zero frames; `MAX_DURATION` discards; file/serial/BLE parity; no state leaks; the stall watchdog. | 56 |
| [tests/test_faults.py](tests/test_faults.py) | The three fault modes through logger and streamer; byte identity with faults off; the accounting identity; the CLI. | 55 |
| [tests/test_parse_frame.py](tests/test_parse_frame.py) | The frame codec's rejection paths (frame spec section 8). | 9 |
| [tests/test_train_real.py](tests/test_train_real.py) | The splits (no overlap, order, sizes, contiguous blocks); identity gains make C equal B; the shipped representation reproduces the bake-off frame and raw counts still give the pre-switch figure; `scripts/train_real.py` end to end. | 28 |

`tests/test_infer_live.py` needs `data/sim/features_sessions.csv`, which
`python scripts/bakeoff.py` builds; without it one check reports the missing
file and the fix.

## Open items

- **BLE stalls when the board is powered from the host PC's USB cable.** The
  stage-14 bench passed on all four runs (hardware section), but only with the
  board on battery for the BLE two; two attempts on PC power dropped the link
  ~210 ms after the connection-parameter request. Three variables changed
  before the passing run, so the cause is narrowed to the Windows-side central,
  not identified. Powering from a battery is the workaround.
- **One session per class.** Four minutes of real data, one 60 s trial per
  activity, from one subject on one day, walking a figure-8 in a small space.
  No per-session split is possible, so any real-data accuracy is a
  within-session number with the leakage that implies.
- **Weight-shift was never collected**, although the collection plan listed it.
- **The gain match extrapolates** above 824 counts, which is most loaded
  walking frames, and does not hold below about 5 N, where the channels'
  activation thresholds diverge (s4 read 0 counts at 2.58 N while s5 read 239
  at 2.49 N).
- **s4's activation threshold** makes the CoP a five-sensor centroid on half
  the moving frames, with a 33.6–36.7 mm lateral bias. Its zeros are
  below-threshold readings, never imputed or dropped.
- **Sensor coordinates carry ±15 mm** on a 91 mm wide insole; a jig would
  tighten them.
- **Detector thresholds are simulator-swept.** `T_ON`, `T_OFF`, `MIN_DURATION`
  and `GAP_MERGE` were chosen against streams whose constants were co-evolved
  with them; only `MAX_DURATION` has been set from real data, and only real
  data can exercise it.
- **The features see conductance (representation B), not the gain match.**
  Stage 20 chose it on the real captures by a pre-registered rule; the
  gain-matched representation C scored no better there while extrapolating
  on most loaded frames, so the gain match drives the extrapolation counter
  only. The margin between the representations is a couple of stances in
  113; this is a decision, not a finding.
- **Firmware, left as is:** `Serial.printf` from core 0 can interleave with
  frame writes from core 1 (rare, MTU transition only); the BLE gather loop's
  size-guard discard neither increments `bleDropped` nor advances the cursor
  (cannot trigger at current sizes).
- **Untethered BLE walking** needs a USB power bank and a rerun of the bench.
- **`force_n` / `sigma_F` in the calibration manifest** bake estimator choices
  (g = 9.81, range/√12) into an append-only file, and the recorded ~2 % force
  uncertainty understates the real one: the operator reading the scale
  dominates.
