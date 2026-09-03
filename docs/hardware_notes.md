# Hardware notes

Everything the bench and the board taught, in one place. Each number names
the file in this tree that records it; a figure marked **session note** was
measured at the bench but is not reproducible from anything committed, so it
is guidance, not evidence.

## Board and toolchain

- ESP32-S3-DevKitC-1 (WROOM-1), Arduino-ESP32 core, sketch
  `firmware/insole/insole.ino` with `sample.h` (an inactive helper header kept
  so the local-helpers switch still compiles). The sketch folder name must
  equal the sketch name.
- Set *Tools ▸ USB CDC On Boot: Enabled* before flashing; without it the board
  emits nothing on the USB port (session note). `Serial.begin(921600)`; the
  baud is ignored on native USB CDC (`insole.ino`, `setup()`).
- The QT Py ESP32-S3 was rejected: two ADC1 pins only, and ADC2 is unusable
  while the radio is on (`docs/frame_spec.md`, appendix).
- BLE is the Arduino core's Bluedroid-style API (the `BLEDevice`/`BLEServer`
  includes at the top of `insole.ino`) running over the stack the core ships;
  the BLE additions are bracketed `// ===== BLE ADDED =====` ... `END` so the
  serial-only sketch can be recovered by deleting them.

## Pins, dividers, sampling

- `PINS = {4, 5, 6, 7, 8, 3}` carry s0..s5, all ADC1 (`insole.ino`). GPIO3 is
  a JTAG strapping pin: fine as an ADC input, never add a pull-up or
  pull-down (comment in `insole.ino`).
- 1 kΩ divider on every channel (`insole/calibration.py`, module docstring).
- 12-bit reads, `ADC_11db` attenuation, `OVERSAMPLE = 8` averaged per sample,
  `PERIOD_US = 10000` (100 Hz) on an absolute-deadline scheduler
  (`nextDueUs += PERIOD_US`), timestamp `esp_timer_get_time()` relative to a
  latched t0 (`insole.ino`; `docs/frame_spec.md` section 3).
- The checksum accumulates in a `uint64_t` (`insole.ino`, `buildFrameLine`).
  The spec's arithmetic only needs 32 bits; the 64-bit accumulator is what the
  sketch settled on after an earlier accumulator width diverged from the host
  on long runs (session note).
- The serial drain never blocks and counts truncated lines in `frameTruncs`
  (`insole.ino`); BLE frames go through a FreeRTOS queue with a zero-timeout
  producer and drop-oldest on overflow (`bleEnqueue`), `FRAMES_PER_NOTIFY = 3`,
  an MTU gate at `MIN_USABLE_MTU = 100`, `xQueueReset` on disconnect, and a
  `# ble ...` status line once per second that the host counts as `status`.

## Sensor placement

`insole/detector.py`, `SENSOR_MM`, measured 2026-09-02 on the right insole
lying top-up, x from the medial edge, y from the heel end, insole 274 × 91 mm
(`data/real/README.md` gives the same table in inches):

| sensor | x (mm) | y (mm) | anatomy |
| --- | --- | --- | --- |
| s0 | 33.0 | 50.8 | heel, medial |
| s1 | 50.8 | 50.8 | heel, lateral |
| s2 | 76.2 | 152.4 | lateral midfoot |
| s3 | 81.3 | 185.4 | 5th metatarsal head |
| s4 | 25.4 | 203.2 | 1st metatarsal head |
| s5 | 25.4 | 254.0 | hallux |

Two measurement passes disagreed by 12–22 mm, so treat every position as
±15 mm (`insole/detector.py`). An older 295 × 74 mm figure was wrong in both
axes. Mean activation order in real walking, s1 → s0 → s2 → s3 → s4 → s5 on
peak timing, matches anatomy, so no channel pair is swapped
(`scripts/sim_vs_real.py` D5). `heatmap.FOOT_OUTLINE` is decorative; the
authoritative boundary is the rectangle.

## Flashing and capture gotchas

- Close the Arduino Serial Monitor before running any Python capture: one
  program per port, or Windows raises `PermissionError` (session note).
- `--port` overrides the default in `insole/read_serial.py`; nothing
  machine-local is committed. `--source ble` needs no port.
- A capture that goes quiet for 3 s while the link stays up is a failure, not
  a short file: `STALL_S` in `insole/read_serial.py`.

## Noise floor

Session note (stage 4, `scripts/capture_noise.py` + `scripts/noise_stats.py`
with sensors attached and unloaded): worst-case peak-to-peak 13 counts on s5,
so `T_OFF = 450` has about 35× margin. The "samples beyond 5σ" column misleads
at σ ≈ 0.15 counts, because single-count quantisation flicker exceeds 5σ; it
is not a fault. `scripts/capture_noise.py` still carries the UART-bridge port
and baud it was written for; edit them before reuse.

## Calibration rig and procedure

- What an FSR reports depends on the engaged contact area and the local
  pressure distribution, not on total force alone. The working indentor is a
  four-layer glued card puck (stiffness ∝ t³) loaded at its centre by a
  small-diameter nub; the replicate gate was two consecutive presses within
  about 15 counts at about 500 g (session note).
- Force comes from a scale read as a min–max interval; `g_min` and `g_max`
  are stored raw in `cal_data/calibration_manifest.csv` and `force_n`,
  `sigma_force_n` derived from them (`scripts/capture_calibration.py`,
  `append_manifest`). The estimator is the interval midpoint at g = 9.81 with
  a range/√12 spread (session note). That σ, about 2 %, understates the real
  force uncertainty by roughly 10×: two presses at the same actual force were
  logged 586 g apart, so the operator reading the scale dominates (session
  note). Both columns are baked into an append-only file.
- One trial per press, `cal_s{N}_t{K}.csv` plus one manifest row
  (`scripts/capture_calibration.py`); 42 trials and the manifest are tracked
  under `cal_data/`.
- The shipped result is the single-point relative gain match: the six
  sensors' trial 6, all at 11.5–12.0 N after 39–42 min rest, selected by
  `rest_min >= 35` and `11.4 <= force_n <= 12.1` (`models/gain_match.json`,
  `insole/fit_calibration.py`); k = F/x per channel 59.55 / 57.84 / 58.59 /
  75.27 / 52.28 / 57.37 N, corrections 0.9900 / 0.9616 / 0.9741 / 1.2513 /
  0.8692 / 0.9538 (`docs/calibration_notes.md`). Regenerate with
  `python -m insole.fit_calibration`.

## Stress relaxation and the rest schedule

- Under constant applied force the counts fell about 31 % in 76 s and
  recovered with a time constant of about 20 min, reproduced across two bench
  sessions with a clean gradient of count against rest interval
  (`docs/calibration_notes.md`).
- Session notes behind that: the fall was linear at −3.80 counts/s (R² 0.92)
  with the applied force steady to 2.2 % over five presses; a controlled
  recovery, rig untouched for 10 min then re-pressed, recovered 35 % raw and
  43 % force-corrected; a single-exponential fit gave τ ≈ 18–23 min, which
  predicted the ~99 % recovery seen across an unrelated 2 h gap; reversible,
  not damage. Scale of the effect: 5.32 → 21.63 N (4× the force) moved a
  sensor 333 counts while drift moved it 256 counts standing still, 77 % of
  the full-range response, so any calibration whose force rises with time
  fits a clock.
- Rest schedule (session note): 5–14 min rests corrupt the middle force
  levels, 17 min and more give valid data, the 39–42 min cycle was the
  cleanest. Level-by-level repair cannot converge: rest-before-trial ran
  monotonically 5.3 → 19.6 min across the six sensors and a monotonic
  response survived only on the two with 17+ min; four short-rest sensors
  read less at 12 N than at 7.6 N. A multi-point absolute calibration would
  need about 40 min × 42 trials, multi-day, hence the single-point match.
- The 40+ manifest rows are that drift dataset and must not be trimmed
  (`docs/calibration_notes.md`).

## Collection protocol

Tethered USB, one 60 s file per activity at 100 Hz, one subject, walking a
figure-8 in a small space, shuffle as feet dragging with short strides and
minimal clearance; the planned weight-shift activity was not collected
(`data/real/README.md`). Files are committed under their as-captured names
(`stand_02.csv` has an underscore, the other three do not). Close the Serial
Monitor first; one program per port (session note).

## Failure modes observed

- **Intermittent channel.** The `_01` set has s0 flat zero in fast and shuffle
  and coming alive mid-walk around seq 205–210 (0 → 12 → 46 → 200): an FSR
  tail without strain relief, which the open-items list had named the likeliest
  mechanical failure under walking. Strain relief was added before `_02`;
  `_01` is failure evidence only (`data/real/README.md`).
- **Correlated single-frame dips.** s1/s2/s3/s5 dip 5–8 % together for
  exactly one frame: a shared-reference or supply glitch, not sensor
  behaviour (`data/real/README.md`).
- **s4's activation threshold.** 0 counts at 2.58 N where s5 read 239 at
  2.49 N; on 52–56 % of moving frames the CoP is a five-sensor centroid biased
  33.6–36.7 mm laterally (`scripts/analyze_real.py` C4/C4b). Below-threshold,
  never imputed.
- **BLE teardown.** The link dropped shortly after connecting with the
  connection parameters the central chose, supervision timeout included; the
  suspected trigger (`insole.ino`, comment above `requestConnParams`). The
  firmware now requests 15–22.5 ms interval, latency 0, 6 s supervision once
  per connection inside the MTU gate (`requestConnParams(conn, 12, 18, 0,
  600)`). Session notes: the symptom was 15 frames over 56 s while the board
  produced about 6000 and the host reported success, firmware counters
  `conns=1 disc=1`; bleak's `disconnected_callback` never fired because the OS
  keeps the GATT object alive while notifications stop, so only the
  inactivity watchdog sees it. Two host bugs found the same day, both fixed
  in the tree: `#` status lines were counted as corruption (valid = 0 while
  the board counted 5804 frames), and the logger exited 0 while printing
  FAIL. Today `#` lines are the `status` counter and every FAIL path returns
  1 (`insole/read_serial.py`).
- **The notebook's timestamps.** `load_sessions` used to overwrite `ts_us`
  with the sample index; the firmware's own ±1 µs jitter exposed it once real
  data arrived, and nothing upstream (firmware, logger, calibration capture)
  ever rewrote `ts_us` (`docs/sim_vs_real.md` C1, `docs/calibration_notes.md`).
- **Firmware, left as is.** `Serial.printf` from core 0 can interleave with
  frame writes from core 1 (rare, MTU transition only); the BLE gather loop's
  size-guard discard neither increments `bleDropped` nor advances the cursor
  (cannot trigger at current sizes). Both listed in the README's open items.

## What the live bench must show

The stage-14 bench has not been run in this tree. Commands and pass criteria
are in the README's hardware section: 60 s over serial and 60 s over BLE
through `python -m insole.read_serial`, then `python -m insole.infer_live` on
each; about 6000 valid frames per capture, `malformed=0 bad_checksum=0
seq_breaks=0 timing_breaks=0 resets=0` over serial, `loss` at or under 2 %
and `timing_breaks=0` over BLE, exit 0 everywhere, and identical counters
between logger and streamer. For reference, the session that closed stage 14
reported serial 6000 frames / 60 s with zero faults and BLE 6033 frames /
60.3 s at 100.05 Hz with zero faults, 2190 notifications and 2 drops (session
note); the bench is what turns that into evidence.
