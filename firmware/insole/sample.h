#pragma once

// INACTIVE. insole.ino sets USE_LOCAL_SAMPLE_HELPERS 1, so its
// `#include "sample.h"` sits behind `#if !USE_LOCAL_SAMPLE_HELPERS` and never
// fires; the live sampleChannel()/buildFrameLine() are in the sketch. Kept
// rather than deleted so flipping that switch back to 0 still compiles.

#include <stdint.h>

// One instant of capture: sequence number, timestamp, six channel readings.
// Lives in a header rather than the sketch because the Arduino builder injects
// auto-generated function prototypes after the last #include, and three
// functions take a Sample& — the type must be visible before that point.
struct Sample {
    uint16_t seq;
    int64_t  t_us;
    uint16_t v[6];
};