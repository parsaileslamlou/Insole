# Project log

The build record and the full audit trail: what was built in what order, what
this project accepts and will not fix, and what was evaluated and resolved.

The [README](../README.md) carries the summary of all of this. Nothing here is
a to-do list; every entry is closed.

## Build stages


| Stage | What | Where | Status |
| --- | --- | --- | --- |
| 1 | Wire-format specification | [frame_spec.md](frame_spec.md) | done |
| 2 | Frame codec + gait simulator | `insole/gait_gen.py` | done |
| 3 | Firmware: six-channel 100 Hz sampler | `firmware/insole/` | done |
| 4 | Host logger, noise-floor utilities | `insole/read_serial.py`, `scripts/capture_noise.py`, `scripts/noise_stats.py` | done |
| 5 | Notebook pipeline: load, clean, window | `notebooks/insole.ipynb` | done |
| 6 | Stance detection | `insole/detector.py` | done |
| 7 | Per-stance features and centre of pressure | `insole/features.py` | done |
| 8 | Pressure heatmap and stance animation | `insole/heatmap.py` | done |
| 9 | Simulator regression fixtures and detector tests | `data/sim/`, `tests/test_stances.py` | done |
| 10 | Multi-session sim data, from-scratch LDA/QDA | `insole/make_sessions.py`, `insole/discriminant.py` | done |
| 11 | Model bake-off | `scripts/bakeoff.py`, [bakeoff.md](bakeoff.md) | done |
| 12 | Calibration: single-point relative gain match | `insole/calibration.py`, `cal_data/`, `models/gain_match.json` | done |
| 13 | Real captures and sim-vs-real analysis | `data/real/`, `scripts/analyze_real.py`, [sim_vs_real.md](sim_vs_real.md) | done; the planned weight-shift activity was not collected |
| 14 | Streaming inference over serial and BLE | `insole/infer_live.py`, firmware BLE | done 2026-09-03; live bench run on the board, four of four captures pass (numbers below) |
| 15 | Fault injection and robustness | `insole/gait_gen.py` fault modes, `tests/test_faults.py` | done |
| 16 | Repository consolidation, the README | package layout | done |
| 17 | Demo notebook | [notebooks/demo.ipynb](../notebooks/demo.ipynb), `figures/demo/` | done |
| 18 | Writeup | [writeup.md](writeup.md) | done |
| 19 | Final pass, `run_demo.sh`, hardware notes | [run_demo.sh](../run_demo.sh), [hardware_notes.md](hardware_notes.md) | done |
| 20 | Classifier trained on the real captures | `scripts/train_real.py`, [real_results.md](real_results.md) | done; with two sessions per class the headline is leave-one-session-out, every stance tested out of its own session |
| 21 | Second session per class, generalised stance guard, closeout | `data/real/*_03.csv`, `scripts/train_real.py`, `tests/`, the README | done 2026-09-03 |

This record is final. The two lists below replace the open-items list: what
this project accepts and will not fix, and what was evaluated and resolved.
Neither is a to-do list.

## Accepted limitations

Each of these is understood, measured where it can be measured, and left as
it stands. The sentence after each says why it is accepted rather than fixed.

- **The 2026-09-03 bench exit codes were derived, not recorded.** The four
  runs were typed by hand and their logs end at `read_serial`'s summary line,
  so the `exit` column was reconstructed by feeding each log's counters back
  through `read_serial.exit_code` (a pure function of those counters, 0 for
  all four); it cannot see an exception, a signal, or a failed write after the
  summary printed (`docs/hardware_notes.md`). *Accepted because re-recording
  them means re-running the bench on hardware, and the reconstruction is sound
  for everything the counters describe — `bench_capture.sh` now records the
  status directly, so the gap is closed forward and only these four runs carry
  it.*
- **BLE stalls when the board is powered from the host PC's USB cable.** The
  stage-14 bench passed on all four runs, but the BLE two only on battery; two
  attempts on PC power dropped the link ~210 ms after the connection-parameter
  request. Three variables changed before the passing run, so the cause is
  narrowed to the Windows-side central, not identified. *Accepted because
  isolating it needs a second central and a protocol capture on hardware this
  project no longer has time on, and a battery is a complete workaround for
  the only use the insole has.*
- **Detector thresholds are simulator-swept.** `T_ON`, `T_OFF`,
  `MIN_DURATION` and `GAP_MERGE` were chosen against streams whose constants
  were co-evolved with them; only `MAX_DURATION` was set from real data.
  *Accepted because re-sweeping them honestly needs labelled real stances from
  more than one subject, which this dataset cannot supply — and the circularity
  is disclosed rather than hidden, which is the most this data supports.*
- **One subject, two sessions per class.** Two sessions is the minimum that
  makes a per-session split possible, not a comfortable margin: with two folds
  one odd session moves the number a lot, and it is still one subject on one
  figure-8 path. Weight-shift was never collected, although the collection plan
  listed it. *Accepted because the fix is a data-collection campaign, not a code
  change, and every number in this repository is already reported against the
  session-disjoint split with its denominator named.*
- **The calibration gain match never reaches the classifier.** The features see
  conductance (representation B); the gain-matched representation C scores the
  same as B on the two-session data (0.5982 both) while extrapolating above 824
  counts on most loaded walking frames, and does not hold below about 5 N.
  *Accepted because shipping C would be a default change plus a refit for a
  move of at most two stances in 224 — a change this data cannot show to be an
  improvement — so the gain match drives the extrapolation counter only.*
- **Every pooled interval is a lower bound on the uncertainty, not a
  confidence interval for a new session.** The Wilson intervals treat 224
  stances from 2 sessions of 1 subject as 224 independent observations; they
  are positively correlated, so the true interval is wider. *Accepted because
  correcting it needs the between-session variance, and two sessions estimate
  that from two points — one degree of freedom, which is not an estimate; the
  per-fold intervals are reported beside every pooled one for exactly this
  reason.*
- **The pre-registered rule now prefers raw counts, by a margin finer than the
  data it is deciding on.** It picks A (raw) over B (conductance) by two
  stances in 224 — 0.9 % — and the two cells' Wilson intervals, [0.5419,
  0.6688] for A and [0.5329, 0.6602] for B, overlap over 93 % of their length.
  **A and B are statistically indistinguishable on this data. B remains
  shipped and is FROZEN**, retained as a pre-existing freeze taken at stage 20,
  not as this rule's verdict — any statement that the rule selected B is wrong.
  *Accepted because switching the shipped representation moves the sim bake-off
  frame, the sim-trained models and the streaming path together, for a change
  this data cannot show to be an improvement.*
- **s4's activation threshold makes the CoP a five-sensor centroid.** Over all
  moving frames of the `_02` captures s4 reads 0 on 46–56 % of them, a
  common-substitute bias of 33.6–36.7 mm (`scripts/analyze_real.py` C4/C4b);
  inside kept stances it is 20–35 % of frames and a 12–14 mm shift
  (`docs/real_results.md` section 7). Its zeros are below-threshold readings,
  never imputed or dropped. *Accepted because it is a property of the sensor at
  low load, not a bug in the pipeline, and the cost is measured in millimetres
  in two places rather than corrected away.*
- **Sensor coordinates carry ±15 mm** on a 91 mm wide insole. *Accepted because
  tightening them needs a jig and a rebuild of the insole, and the figure is
  quoted beside every CoP number rather than dropped.*
- **Firmware, left as is:** `Serial.printf` from core 0 can interleave with
  frame writes from core 1 (rare, MTU transition only); the BLE gather loop's
  size-guard discard neither increments `bleDropped` nor advances the cursor.
  *Accepted because neither can trigger at current frame sizes, and both are
  recorded here rather than patched blind on a board that is no longer on the
  bench.*
- **`force_n` / `sigma_F` in the calibration manifest** bake estimator choices
  (g = 9.81, range/√12) into an append-only file, and the recorded ~2 % force
  uncertainty understates the real one: the operator reading the scale
  dominates. *Accepted because the file is append-only by design and the raw
  counts it derives from are all still there, so a better estimator can be
  applied downstream without rewriting the manifest.*

## Resolved

Both branches that were parked on the other machine have been swept. Neither
is on this remote; both are archived in a local git bundle,
`insole_archive_branches.bundle`, which `git bundle verify` reports as
carrying their complete history.

- **`cal-wls-local` — evaluated and rejected.** Its `1/sigma_x^2`-weighted
  calibration fit was measured against the shipped unweighted OLS fit and not
  adopted: it shifts the fitted slope 2.5–7.1 % on five of six channels,
  changes no `MIN_R2` flag (every channel the shipped fit marks `poor_fit`
  stays `poor_fit`), and the information ratio that motivates it is 1.9–2.3x
  on the captures that exist rather than the 19.9x quoted for a hypothetical
  load ladder. Both estimators, and the reasoning, are written up in
  `docs/calibration_notes.md`.
- **`wip/ble-transport` — landed.** Its BLE reader-thread failure fix is now
  in `insole/read_serial.py`: the reader's exception is parked and re-raised in
  the consumer once every queued line has been yielded, so a dead radio
  produces a traceback and a nonzero exit instead of a short capture that looks
  clean. `tests/test_ble_transport.py` came with it and covers both that path
  and the batched-arrival timing check.

## Reference detail

Detail moved out of the README when it was cut to a summary. Nothing here is
superseded; it is kept out of the front page because it serves someone already
working in the repository, not someone deciding whether to read it.

### Capture, on hardware

Close the Arduino Serial Monitor first (one program per port).
`--port` overrides the default in `insole/read_serial.py`; nothing
machine-local is committed.

```
python -m insole.read_serial --source serial --port COM13 --duration 60 bench_serial.csv
python -m insole.read_serial --source ble --duration 60 bench_ble.csv
python -m insole.infer_live --source serial --port COM13 --duration 60 --out bench_serial_preds.csv
python -m insole.infer_live --source ble --duration 60 --out bench_ble_preds.csv
```

### Exit codes and what the fault handling does not cover

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


### Analysis notebook

[notebooks/insole.ipynb](../notebooks/insole.ipynb) runs on the saved CSVs and is
written for Colab; its first cell clones the repository, installs the package,
and rebuilds the simulated CSVs from the committed streams. Locally, execute it
from the repository root. Stages: load and clean (with `ts_us` exactly as the
firmware recorded it; an earlier version overwrote it with the sample index,
which hid timing jitter and was caught only when real data arrived), window,
stance detection, scoring against `gait_gen.true_stances`, features, and the
walk-versus-fast feature plots. The pressure heatmap between the six sensors
is inverse-distance-weighted interpolation, not measurement, and the render
says so.


### Test files, one by one


| File | Guards | PASS lines |
| --- | --- | --- |
| [tests/test_stances.py](../tests/test_stances.py) | Detector counts against `gait_gen.true_stances` on the five sim streams; `merge_close`; the real stance counts, `_02` 0 / 35 / 48 / 30 and `_03` 0 / 32 / 45 / 34. | 23 |
| [tests/test_geometry.py](../tests/test_geometry.py) | `SENSOR_COORDS` derived from `SENSOR_MM`; closed-form CoP answers; the documented all-zero `(nan, nan)`. | 51 |
| [tests/test_calibration.py](../tests/test_calibration.py) | Force fit recovery, flags, saturation; the gain match multiplies conductance, not counts; the six shipped corrections. | 55 |
| [tests/test_capture_window.py](../tests/test_capture_window.py) | The capture window is anchored on the first line, not process start. | 13 |
| [tests/test_discriminant.py](../tests/test_discriminant.py) | LDA/QDA against sklearn; Wilson interval; weighted pooled covariance; the `n_k <= p` guard. | 7 |
| [tests/test_infer_live.py](../tests/test_infer_live.py) | Streaming == batch features on every capture; read boundaries; malformed frames; all-zero frames; `MAX_DURATION` discards; file/serial/BLE parity; no state leaks; the stall watchdog; the model's representation drives the feature path and an unhonourable one is refused. | 81 |
| [tests/test_faults.py](../tests/test_faults.py) | The three fault modes through logger and streamer; byte identity with faults off; the accounting identity; the CLI. | 55 |
| [tests/test_parse_frame.py](../tests/test_parse_frame.py) | The frame codec's rejection paths (frame spec section 8). | 9 |
| [tests/test_ble_transport.py](../tests/test_ble_transport.py) | The BLE reader thread's failure reaches the consumer rather than being swallowed; batched notification arrival is not counted as a broken sample clock. Needs no radio and never imports `bleak`. | 2 |
| [tests/test_train_real.py](../tests/test_train_real.py) | The splits (no overlap, order, sizes, contiguous blocks); leave-one-session-out on the two sessions (two folds, every stance once, no session on both sides, the per-fold class counts); identity gains make C equal B; the shipped representation reproduces the bake-off frame and raw counts still give the pre-switch figure; `scripts/train_real.py` end to end, with the split the data on disk allows. | 37 |

Every file's checks print PASS/FAIL lines and return their counts; the
`test_*` wrappers pytest collects assert that nothing failed, so a red check
fails pytest too. The PASS-line column above is the count once
`scripts/bakeoff.py` has run.

**The two counts move independently, and only the script-suite one moves.**
`tests/test_infer_live.py` and `tests/test_train_real.py` read
`data/sim/features_sessions.csv`, which `scripts/bakeoff.py` builds and
`.gitignore` excludes. Without it each file prints one `SKIP` line naming the
command that produces it, and a skipped check counts as neither a pass nor a
failure. The two `SKIP` lines are script-suite checks, not pytest tests:
pytest collects 49 functions and passes 49 either way, because the wrappers
assert only that nothing *failed*.

| Run directly (`python tests/test_*.py`) | Fresh clone | After `python scripts/bakeoff.py` |
| --- | --- | --- |
| `tests/test_infer_live.py` | 80 PASS, 1 SKIP | 81 PASS |
| `tests/test_train_real.py` | 33 PASS, 1 SKIP | 37 PASS |
| the other eight files | 215 PASS | 215 PASS |
| **total** | **328 PASS, 0 FAIL, 2 SKIP** | **333 PASS, 0 FAIL, 0 SKIP** |
| `python -m pytest -q` | 49 passed | 49 passed |

The two `SKIP` lines gate five checks between them, not two: the
`test_train_real.py` skip stands in front of four. `scripts/fit_model.py` is
not needed for either — `models/model_lda.json` is committed — so
`python scripts/bakeoff.py` alone takes a fresh clone from 328 to 333.
**A fresh clone is green before anything is generated**, and a missing
generated artifact and a real failure are not reported as the same thing.

