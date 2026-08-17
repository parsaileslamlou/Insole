# Serial Frame Specification

**Version:** 1.0
**Applies to:** ESP32-S3 firmware, host logger (`host/read_serial.py`), gait simulator (`sim/gait_gen.py`)

## 1. Scope

This document defines the wire format for pressure data sent from the ESP32-S3 to the host.

Three components must agree on this format exactly: the firmware that produces frames, the host logger that consumes them, and the simulator that produces synthetic frames for hardware-free development. Any disagreement between them produces validation failures with no useful error message, so the format is specified here rather than being implied by the code.

The spec defines frame *contents*, not the transport. v1 uses USB CDC serial. A different transport (e.g. BLE) would be a new line-source implementation on the host, with no change to this document.

## 2. Frame format

One frame is one newline-terminated ASCII line:

```
SYNC,SEQ,TIMESTAMP,S0,S1,S2,S3,S4,S5,CHECKSUM\n
```

Ten comma-separated fields. Terminator is LF (`0x0A`), not CRLF. All numeric fields are plain decimal with no padding, no leading zeros, and no whitespace (`300`, not `0300` or ` 300`).

Example frame:

```
INS,41,152300,2048,1900,300,150,2200,3000,147
```

## 3. Fields

| # | Field | Type | Range | Description |
|---|-------|------|-------|-------------|
| 0 | SYNC | string literal | `INS` | Line-type tag |
| 1 | SEQ | uint16 | 0–65535, wraps | Frame counter, increments once per frame |
| 2 | TIMESTAMP | int64 | µs since capture start | Sample time, generated on the board |
| 3–8 | S0–S5 | uint16 | 0–4095 | Raw 12-bit ADC counts, one per FSR |
| 9 | CHECKSUM | uint8 | 0–255 | Integrity check, see §4 |

### Sensor channel order

Channel order is fixed by this document and must not be re-derived elsewhere.

| Channel | Sensor position |
|---------|-----------------|
| S0 | Heel (calcaneus) |
| S1 | Lateral midfoot |
| S2 | 1st metatarsal head |
| S3 | 3rd metatarsal head |
| S4 | 5th metatarsal head |
| S5 | Hallux |

Readings are raw ADC counts. Calibration to force units happens on the host, never in firmware, so recalibration does not require a reflash.

### Timestamp

Timestamps are microseconds elapsed since the start of capture, not since boot. Firmware latches `t0` when streaming begins and emits `esp_timer_get_time() - t0`.

`esp_timer_get_time()` is used rather than Arduino's `micros()` because `micros()` truncates the same hardware counter to 32 bits and wraps every 71.6 minutes. `esp_timer_get_time()` returns int64 microseconds.

Microsecond resolution rather than milliseconds: at 100 Hz the sample interval is 10 ms, so millisecond quantisation would impose a ±10% error floor on every Δt, and loading rate is computed as ΔF/Δt. Microseconds reduce that to 0.01%.

## 4. Checksum

Additive sum of the eight integer fields, modulo 256:

```
CHECKSUM = (SEQ + TIMESTAMP + S0 + S1 + S2 + S3 + S4 + S5) % 256
```

SYNC is excluded (it has no integer value). CHECKSUM is excluded (circular).

The sum runs over field *values*, not over the characters of the line. This makes it formatting-invariant: `007` and `7` contribute identically, and a stray space cannot change the result. It also means the firmware sums integers it already holds before any string exists, and the host sums integers its parse gate has already produced.

Firmware must accumulate in a `uint32_t`. Signed overflow is undefined behaviour in C++; unsigned wraparound is defined. Overflow of the accumulator is harmless because 2³² is an exact multiple of 256 (2³² = 256 × 2²⁴), so `(sum % 2³²) % 256` equals `sum % 256`.

**Worked example** — SEQ = 41, TIMESTAMP = 152300, S0–S5 = 2048, 1900, 300, 150, 2200, 3000:

```
41 + 152300 + 2048 + 1900 + 300 + 150 + 2200 + 3000 = 161939
161939 % 256 = 147
```

## 5. Sample rate

100 Hz — one frame every 10,000 µs.

Walking cadence is 1–2 Hz and most ground-reaction-force content sits below 10 Hz. The fastest feature of interest is the heel-strike transient, which rises over roughly 10–30 ms and therefore carries content to somewhere around 25 Hz. Nyquist puts the reconstruction floor near 50–60 Hz, but the pipeline is not reconstructing a waveform; it is timing events (heel strike, toe off) and estimating slopes (loading rate). Event-timing resolution at 60 Hz is ~17 ms, which is coarse against a 20 ms transient, so the rate is set at 3–5× f_max rather than 2×.

100 Hz gives 10 ms event resolution, roughly 60–100 samples per step, and 40–60 per stance phase. Higher rates were rejected as pure cost: at 1 kHz the additional samples carry no new information, while file size, link load, and validation work all grow 10×.

Firmware headroom at 100 Hz is large. Six `analogRead()` calls cost roughly 300–600 µs against a 10,000 µs budget on a 240 MHz core, leaving room for oversample-and-average during ADC sampling.

## 6. Design notes

### Text CSV rather than packed binary

Both were considered. A packed binary frame carrying these fields (1-byte sync, uint16 seq, int64 timestamp, 6 × uint16, 1-byte checksum) is 24 bytes. Measured CSV output from the simulator averages 38.7 bytes per frame across a 60-second walk. At 100 Hz that is 2.4 kB/s versus 3.9 kB/s. The link is USB CDC at 12 Mbit/s, so both consume roughly 0.1% of available bandwidth. The efficiency advantage of binary is real and buys nothing at this operating point.

CSV, by contrast, is directly readable at every stage: a capture file can be inspected with `cat`, and a serial terminal shows live sensor values during bring-up. With a binary frame, debugging the format and debugging the decoder become the same task.

CSV also gets framing for free. A newline byte cannot occur inside a decimal number, so `\n` is an unambiguous frame boundary. A binary sync byte offers no such guarantee: `0xAA` can appear inside a 12-bit reading (`0x0AAA` contains it), so every candidate boundary must be confirmed against the checksum. The practical consequence is resynchronisation behaviour — a lost byte in a CSV stream corrupts exactly one line, and the next newline restores alignment, whereas a lost byte in a binary stream desynchronises the parser until a sync search succeeds.

Binary encoding is a candidate for a future revision if the data rate ever grows to where per-frame bytes matter. The field set would not change.

### Why a sync token, given that `\n` already delimits frames

In this format the newline handles boundary recovery, so SYNC is not the resynchronisation mechanism. It earns its place for two other reasons.

First, it rejects tail fragments. A fragment that has lost its head carries no `INS` prefix and fails the sync gate, where it might otherwise present a plausible field count and parse cleanly.

Second, it makes the stream multiplexable. Once firmware emits anything besides data frames on the same wire (`LOG,...`, `ERR,...`), every line must declare its own type. A leading tag makes each line self-describing; without one, the parser would be inferring line type from field count, which is a heuristic rather than a contract.

The token is deliberately non-numeric. If it ever appears in a numeric column through misalignment, `int()` raises immediately rather than the value being silently accepted as data.

### Why sum-mod-256 rather than CRC

The channel is a short wired USB link — no radio, no long cable, no burst-error environment. The realistic failure mode is a dropped or truncated frame caused by a buffer overrun or a host stall, not the subtle multi-byte corruption that CRCs exist to catch. Truncation is caught by the checksum together with the field-count gate.

An additive sum is also verifiable by hand during debugging, which a CRC is not.

CRC-8 is a candidate for a future revision. It would remove the blind spots listed in §7 at the cost of a 256-entry lookup table, with the same compute-over-payload / compare-on-host structure.

### Why 16-bit SEQ

An 8-bit counter wraps every 2.56 s at 100 Hz, which makes a dropout of exactly 256 frames (or any multiple of 256) invisible in the sequence numbers. 16 bits pushes that failure mode out to an 11-minute dropout, which the timestamp would independently flag. The cost is roughly three characters per frame.

## 7. Known limitations

The checksum is blind to any error that leaves the sum unchanged:

- **Field reordering.** The same values in a different order produce the same sum.
- **Compensating errors.** One field +1 and another −1 cancel.
- **Coincidental collision.** Two distinct frames land on the same checksum byte roughly 1 time in 256.

These are accepted for v1 for the reasons given in §6. Fields do not spontaneously reorder on a wired link, and a ~0.4% escape rate on an already-rare corruption event is an acceptable price for a checksum that can be verified by hand.

The timestamp is generated on the board, so the pipeline is insensitive to transport jitter and to batched delivery. It is not insensitive to *loss*: dropped frames leave gaps in the time axis, which breaks anything that assumes uniform sample spacing (windowing, impulse integration, loading-rate estimation). SEQ exists to make that loss detectable rather than silent; the handling policy lives in the ingestion stage.

## 8. Host-side validation

The host applies these gates in order, cheapest first, so that a line of boot text is rejected before any arithmetic runs on it:

1. **Field count** — must be exactly 10.
2. **Sync** — field 0 must equal `INS`.
3. **Integer parse** — fields 1–9 must all parse as integers. Field 0 is skipped; it is not numeric.
4. **Checksum** — recompute over fields 1–8 and compare against field 9.
5. **Sequence** — compare against the previous SEQ and count gaps.

The checksum is not redundant with the gates above it. A single bit-flip in a sensor reading (`2048` → `2049`) preserves the field count, preserves the sync token, and still parses cleanly as an integer; every gate except the checksum accepts it. The same holds for a line truncated *inside* its final field — `...,3000,147` cut to `...,3000,14` still has ten fields and still parses. Only the checksum rejects these.

Truncation that removes a whole field is caught earlier, by the field count. The two gates cover different cuts.

Counters reported at exit: valid, malformed, empty, bad checksum, dropped.

## 9. Open items

- ~~ADC1 pin → sensor mapping (six of GPIO1–GPIO10), pending hardware assembly.~~ Resolved: soldered as GPIO 4, 5, 6, 7, 8, 3 for channels s0–s5, in the §3 order. `PINS` in the sketch is the record of that mapping.
- Physical FSR placement to be confirmed against the channel order in §3. Exact 2D coordinates are needed later for centre-of-pressure and heatmap work, and will live in a single shared coordinate table.

## Appendix: board selection

The QT Py ESP32-S3 was evaluated and rejected. It breaks out four analog pins (A0–A3), of which only two (A2, A3) sit on ADC1. This design requires six ADC1 channels; ADC2 is unusable when WiFi is active.

The ESP32-S3-DevKitC-1 (WROOM-1) breaks out GPIO1–GPIO10, which map to ADC1_CH0–CH9, giving ten usable ADC1 channels.