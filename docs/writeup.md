# Writeup

**Problem.** Six force-sensitive resistors under a right insole, an ESP32-S3
sampling them at 100 Hz, and one question: can a per-stance centre-of-pressure
signal tell walking, fast walking and shuffling apart?

**Architecture.** Firmware frames each sample as
`INS,SEQ,TS_US,S0..S5,CHECKSUM` (`docs/frame_spec.md`) over USB serial or BLE.
The host logger validates and counts; a hysteresis detector segments stances
on total force; seven per-stance features, two of them centre-of-pressure,
feed a from-scratch LDA/QDA. `insole/infer_live.py` runs the same state
machine one frame at a time, bit-identical to batch.

**Strategy.** I wrote the frame spec first, then a simulator emitting the same
frames, so every stage had a stub behind the hardware's interface before any
board. Adversarial streams shaped the detector, seeded sessions trained the
bake-off, and injected faults (drops, bad checksums, a reboot) hardened
logger and streamer. The cost is circularity: the generator's
constants and the detector thresholds were co-evolved, so nothing measured on
simulation is evidence about hardware. The real captures were the first
non-circular test.

**Results.** On the session-disjoint simulated split (270 stances, floor
0.4296) CoP-only LDA scores 0.9185 and QDA 0.9296 (`scripts/bakeoff.py`). The
same models on the 224 real stances score 0.3795 and 0.3438, below the 0.4152
floor. Retrained on the real captures, two sessions per class, every stance
tested out of its own session, the CoP-only headline is QDA on raw counts,
0.6071 [0.5419, 0.6688] on n = 224; the shipped conductance model scores
0.5982, two stances away and statistically indistinguishable from it. Adding
contact time reaches 0.7578 [0.6976, 0.8094] on n = 223 — one `shuffle_03`
stance starts at frame 0, its loading rate undefined, dropped not imputed —
and that cell rides on cadence, not the CoP. **Every pooled interval here is a
lower bound on the uncertainty**: it treats 224 correlated stances from two
sessions of one subject as 224 independent observations, so the true interval
is wider by an amount two sessions cannot estimate (`docs/real_results.md`).
Walk recall is 0.209: 39 of 67 walk stances are called fast — the simulator's
own prediction, confirmed on hardware. Fast and walk are one gait at two
cadences, so with cadence stripped the CoP cannot separate them; shuffle is
what it finds.

**What hardware changed.** The FSRs relax: counts fell about 31 % in 76 s
under constant load, recovering over ~20 min (`docs/calibration_notes.md`), so
a multi-point absolute calibration would fit a clock; what shipped is a
single-point relative gain match at ~12 N in conductance space. It drives the
extrapolation counter only, never the features. The
simulator-derived `MAX_DURATION` of 120 frames did not transfer: real contacts
run 84–164 frames, and at 120 the detector kept 18 of 35 walk and 2 of 30
shuffle contacts (`scripts/sweep_max_duration.py`); 200 keeps all; over-ceiling
runs are discarded, not clipped. s4 has the highest activation
threshold of the six, so on 52–56 % of moving frames the CoP is a five-sensor
centroid biased 33.6–36.7 mm laterally on a 91 mm insole. The first dataset lost
its heel channel to an unrelieved FSR tail and is kept as failure evidence
(`data/real/README.md`). BLE dropped the link seconds after
connecting; the firmware now requests its own connection parameters, and the
60 s bench runs then passed over serial and BLE with zero faults, the BLE ones
only on battery (`data/bench/`). And the notebook had overwritten the board's
timestamps with the sample index — a leak only real jitter exposed.

**Five sentences.** I built a six-sensor smart insole and the host pipeline
that logs, segments, extracts features and classifies gait. I wrote the wire
spec first and a simulator behind the same seam as the board, so every stage
was built, tested and fault-injected before hardware arrived. Reality
corrected me: the sensors relax by a third within a minute, my stance ceiling
cut real walking in half, and one channel rarely turns on. On simulation the
classifier scores 0.93; retrained on real data and tested out of session,
0.61, and it calls walking fast 39 times in 67, the confusion the
simulator predicted once cadence is removed. The limits are honest: one
subject, two sessions per class, ±15 mm sensor positions, intervals that are
floors rather than ranges, and a gain match that never reaches the classifier.
