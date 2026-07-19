#pragma once

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