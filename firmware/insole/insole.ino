// 6-channel FSR sampler for the ESP32-S3.
//
// Samples six force-sensitive resistors at 100 Hz and streams one CSV frame per
// sample over USB CDC. Readings are raw ADC counts; calibration to force units
// is done on the host so that recalibration does not require a reflash.
//
// Frame format (see framespec.md for the full contract):
//
//   INS,<seq>,<ts_us>,<s0>,<s1>,<s2>,<s3>,<s4>,<s5>,<cksum>\n
//
//   seq    uint16, wraps at 65536
//   ts_us  int64, microseconds since capture start
//   s0-s5  raw 12-bit ADC counts, 0-4095
//   cksum  (seq + ts_us + s0 + ... + s5) % 256

#include <Arduino.h>
#include <esp_timer.h>
#include "sample.h"

// ---------------------------------------------------------------- config

// ADC1 pins, one per channel, in the order given by framespec.md section 3.
// TODO: confirm against the assembled hardware.
const uint8_t PINS[6]    = {1, 2, 4, 5, 6, 7};

const int     OVERSAMPLE = 8;      // ADC reads averaged per sample
const int64_t PERIOD_US  = 10000;  // 100 Hz
const int     BUF_N      = 32;     // ring buffer slots

// Set to false once sensors are attached. The stub lets the full pipeline
// (framing, serial, host logger) be exercised with no hardware present.
const bool    USE_STUB   = true;

// ---------------------------------------------------------------- sampling

uint16_t readChannelReal(uint8_t i) {
    uint32_t sum = 0;
    for (int k = 0; k < OVERSAMPLE; k++) {
        sum += analogRead(PINS[i]);
    }
    return (uint16_t)(sum / OVERSAMPLE);
}

// Deterministic 1 Hz sawtooth, phase-shifted per channel.
uint16_t readChannelStub(uint8_t i) {
    int64_t  ms    = esp_timer_get_time() / 1000;
    uint32_t phase = (uint32_t)((ms + i * 167) % 1000);
    return (uint16_t)(phase * 4095 / 999);
}

uint16_t readChannel(uint8_t i) {
    return USE_STUB ? readChannelStub(i) : readChannelReal(i);
}

Sample   ring[BUF_N];
int      head = 0;
int      tail = 0;

int64_t  t0;       // esp_timer value at capture start
uint16_t seq;      // next sequence number to hand out
int64_t  next_us;  // absolute time the next sample is due

// ---------------------------------------------------------------- ring buffer

// On overflow the oldest sample is discarded. Dropped samples are detectable
// on the host as a gap in seq, which is what seq is for.
void bufPush(const Sample& s) {
    int next = (head + 1) % BUF_N;
    if (next == tail) {
        tail = (tail + 1) % BUF_N;
    }
    ring[head] = s;
    head = next;
}

// ---------------------------------------------------------------- framing

// Must match checksum() in gait_gen.py. Accumulating in uint32_t is deliberate:
// unsigned wraparound is defined behaviour, and 2^32 is a multiple of 256, so
// the truncation does not change the result.
uint8_t checksum(const Sample& s) {
    uint32_t sum = 0;
    sum += s.seq;
    sum += (uint32_t)s.t_us;
    for (int i = 0; i < 6; i++) {
        sum += s.v[i];
    }
    return (uint8_t)(sum % 256);
}

int makeFrame(const Sample& s, char* out, size_t n) {
    return snprintf(out, n, "INS,%u,%lld,%u,%u,%u,%u,%u,%u,%u\n",
                    (unsigned)s.seq,
                    (long long)s.t_us,
                    (unsigned)s.v[0], (unsigned)s.v[1], (unsigned)s.v[2],
                    (unsigned)s.v[3], (unsigned)s.v[4], (unsigned)s.v[5],
                    (unsigned)checksum(s));
}

// ---------------------------------------------------------------- setup

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    t0      = esp_timer_get_time();
    seq     = 0;
    next_us = t0;  // loop() advances the deadline before waiting, so frame 1 lands at t0 + PERIOD_US
}

// ---------------------------------------------------------------- loop

void loop() {
    Sample s;
    s.seq  = seq++;
    s.t_us = esp_timer_get_time() - t0;
    for (int i = 0; i < 6; i++) {
        s.v[i] = readChannel(i);
    }
    bufPush(s);

    // Send as much as the link will accept right now, and no more. Writing past
    // availableForWrite() would block and push the sample clock off schedule.
    while (head != tail) {
        char line[80];
        int  len = makeFrame(ring[tail], line, sizeof(line));
        if (Serial.availableForWrite() < len) {
            break;
        }
        Serial.write((const uint8_t*)line, len);
        tail = (tail + 1) % BUF_N;
    }

    // Pace against an absolute deadline rather than a delay, so that per-loop
    // jitter cannot accumulate into sample-rate drift.
    next_us += PERIOD_US;
    while (esp_timer_get_time() < next_us) {
    }
}
