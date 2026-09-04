# Writeup

**Problem.** Six force-sensitive resistors under a right insole, an ESP32-S3
sampling them at 100 Hz, and one question: can a per-stance centre-of-pressure
signal tell walking, fast walking and shuffling apart?

**Architecture.** Firmware frames each sample as
`INS,SEQ,TS_US,S0..S5,CHECKSUM` (`docs/frame_spec.md`) over USB serial or BLE.
The host logger validates and counts; a hysteresis detector segments stances
on total force; seven per-stance features, two of them centre-of-pressure,
feed a from-scratch LDA/QDA. `insole/infer_live.py` runs the same state
machine one frame at a time, bit-identical to the batch path.

**The strategy was a decision.** I wrote the frame spec first, then a
simulator emitting the same frames, so every stage had a stub behind the
hardware's interface before any board. Adversarial simulated streams shaped
the detector, seeded sessions trained the bake-off, and injected link faults
(drops, bad checksums, a mid-capture reboot) hardened logger and streamer. The
cost is circularity: the generator's constants and the detector thresholds
were co-evolved, so nothing measured on simulation is evidence about hardware.
The real captures were the first non-circular test.

**Results.** On the session-disjoint simulated split (270 stances, floor
0.4296) CoP-only LDA scores 0.9185 [0.8797, 0.9456] and QDA 0.9296
[0.8927, 0.9545] (`scripts/bakeoff.py`). The same models on the 224 real
stances score 0.3795 and 0.3438, below the 0.4152 floor. Retrained on the real
captures, two sessions per class, every stance tested out of its own session,
the CoP-only headline under the pre-registered rule is QDA on raw counts,
0.6071 [0.5419, 0.6688] (136/224); the shipped conductance model 0.5982.
Walk recall is 0.209: 39 of 67 walk stances are called fast. That is the
simulator's own prediction, confirmed on hardware. Fast and walk are one gait
at two cadences, so with cadence stripped from the features the CoP cannot
separate them, on the generator or on the foot; shuffle is what it finds.
Adding contact time reaches 0.7578 [0.6976, 0.8094] out of session, on cadence
(`docs/real_results.md`).

**What hardware changed.** The FSRs relax: counts fell about 31 % in 76 s
under constant load and recovered over ~20 min (`docs/calibration_notes.md`),
so a multi-point absolute calibration would fit a clock; what shipped is a
single-point relative gain match at ~12 N in conductance space. It drives the
extrapolation counter only: the classifier sees plain conductance, so the gain
match never reaches the features, and wiring it in would move the headline by
at most two stances in 224. The simulator-derived `MAX_DURATION` of 120 frames did not transfer: real contacts
run 84–164 frames, over-ceiling runs are discarded rather than clipped, and at
120 the detector kept 18 of 35 walk and 2 of 30 shuffle contacts
(`scripts/sweep_max_duration.py`); 200 keeps all. s4 has the highest
activation threshold of the six (0 counts at 2.58 N where s5 read 239), so on
52–56 % of moving frames the CoP is a five-sensor centroid biased 33.6–36.7 mm
laterally on a 91 mm insole (`scripts/analyze_real.py`). The first dataset
lost its heel channel to an unrelieved FSR tail and is kept as failure
evidence only (`data/real/README.md`). BLE dropped the link seconds after
connecting, suspected trigger the host-chosen supervision timeout; the firmware
now requests its own connection parameters, and the 60 s bench runs then passed
over serial and BLE with zero faults, the BLE ones only on battery
(`data/bench/`). And the notebook had overwritten the board's timestamps with
the sample index, a leak only real jitter exposed.

**Five sentences.** I built a six-sensor smart insole and the host pipeline
that logs, segments, extracts features and classifies gait. I wrote the wire
spec first and a simulator behind the same seam as the board, so every stage
was built, tested and fault-injected before hardware arrived. Reality then
corrected me: the sensors relax by a third within a minute, my stance ceiling
cut real walking in half, and one channel rarely turns on. On simulation the
classifier scores 0.93; retrained on real data and tested out of session it
scores 0.61 [0.54, 0.67], and it calls walking fast 39 times in 67 — exactly
the fast-versus-walk confusion the simulator predicted for centre-of-pressure
features once cadence is removed, now confirmed on a foot. The limits are
honest: one subject, two sessions per class, ±15 mm sensor positions, and a
gain match that extrapolates above every load it was built from and never
reaches the classifier.
