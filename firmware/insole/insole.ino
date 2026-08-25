/* =====================================================================
 * insole.ino  —  6ch FSR sampler, 100 Hz, ESP32-S3-DevKitC-1 (WROOM-1)
 *
 * Transports:
 *   USB CDC Serial  — ALWAYS ON, unconditional, unchanged behaviour
 *   BLE NUS notify  — additive, best-effort, never blocks the sampler
 *
 * Frame (unchanged): INS,seq,ts_us,s0,s1,s2,s3,s4,s5,checksum\n
 *   checksum = (sum of the eight integer fields) mod 256
 *
 * Every line added for BLE is bracketed by
 *   // ===== BLE ADDED =====  ...  // ===== END BLE ADDED =====
 * Delete those blocks and the two call sites in setup()/loop() and you are
 * back to the original serial-only sketch.
 * ===================================================================== */

#include <Arduino.h>
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ===== BLE ADDED =====
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>   // If your core errors "BLE2902.h: No such file", delete this
                       // include AND the addDescriptor() line below; core 3.x
                       // attaches the CCCD automatically. A *deprecation warning*
                       // is fine — ignore it.
#include <stdarg.h>     // bleLog() is variadic; see the Serial-ownership note.
#include <atomic>       // bleConnected / bleMtuOk; see the ordering note below.
// ===== END BLE ADDED =====

/* ---------------------------------------------------------------------
 * Existing configuration (unchanged)
 * ------------------------------------------------------------------- */
// s0..s5 -> GPIO. All ADC1 (ADC2 is unusable once the radio is on — and the
// radio IS on now, so this mapping is no longer just a preference).
// NOTE: GPIO3 is a JTAG/strapping pin on the S3. It is fine as an ADC input at
// runtime, but do not hold it externally during reset.
const uint8_t PINS[6]     = {4, 5, 6, 7, 8, 3};
const int     OVERSAMPLE  = 8;
const int64_t PERIOD_US   = 10000;   // 100 Hz
const int     BUF_N       = 32;      // ring buffer slots (serial path)
const bool    USE_STUB    = false;

/* If sample.h already provides an oversampled read and a frame builder, set
 * this to 0 and uncomment the include. Left at 1 so this file compiles and
 * runs standalone. */
#define USE_LOCAL_SAMPLE_HELPERS 1
#if !USE_LOCAL_SAMPLE_HELPERS
  #include "sample.h"
#endif

#define MAX_LINE_LEN 72   // worst-case frame is ~57 bytes + NUL

// buildFrameLine() hit the buffer limit. Not a BLE counter — it belongs to the
// serial framing path, and it is nonzero only if the frame format outgrew
// MAX_LINE_LEN. Surfaced in the 1 Hz status line.
static volatile uint32_t frameTruncs = 0;

// ===== BLE ADDED =====
#define BLE_ENABLED        1
#define BLE_DEVICE_NAME    "INSOLE"
#define FRAMES_PER_NOTIFY  3      // compile-time, per the spec decision
#define MIN_USABLE_MTU     100    // refuse to notify below this
#define BLE_QUEUE_LEN      64     // ~0.64 s of slack at 100 Hz
#define LOG_QUEUE_LEN      4      // core-0 diagnostics awaiting core 1's Serial
#define MTU_WAIT_US        2000000LL  // how long to wait for MTU negotiation

// Nordic UART Service — standard UUIDs, not custom.
#define NUS_SERVICE_UUID "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define NUS_RX_UUID      "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  // central -> us (unused)
#define NUS_TX_UUID      "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  // us -> central (notify)

typedef struct {
  uint8_t len;
  char    text[MAX_LINE_LEN];
} LineMsg;

static QueueHandle_t       bleQueue          = nullptr;
static QueueHandle_t       logQueue          = nullptr;
static BLEServer*          pServer           = nullptr;
static BLECharacteristic*  pTxChar           = nullptr;
/* The two cross-core flags. Relaxed atomics, and the "relaxed" is explicit at
 * every single use -- see the note below for why the default would be wrong.
 *
 * WHY THESE ARE SAFE AS PLAIN volatile TODAY
 * ------------------------------------------
 * Internal SRAM on the S3 is uncached, so there is no cache-coherency problem
 * between the two cores to begin with. A single-word, naturally-aligned load or
 * store cannot tear on this architecture: a reader sees the old value or the
 * new one, never half of each. The only genuine cross-core reader is
 * bleEnqueue() reading bleConnected from core 1 while a BLE callback writes it
 * from core 0, and the worst case there is one frame decided against a flag
 * that flipped a microsecond later -- a frame that bleSkippedNoConn or
 * bleDropped already counts. So the volatile version is not racy in practice.
 *
 * WHY IT CHANGES ANYWAY
 * ---------------------
 * All of that rests on the BLE stack running on core 0, which is true because
 * CONFIG_BT_BLUEDROID_PINNED_TO_CORE defaults to 0. That is a default in
 * somebody else's sdkconfig, invisible from this file, and it can change under
 * us. std::atomic<bool> with memory_order_relaxed compiles to the same load and
 * store instructions on this target -- no barriers, no library calls -- so the
 * guarantee is free and it survives that drift.
 *
 * WHAT RELAXED BUYS, AND WHAT IT DOES NOT
 * ---------------------------------------
 * It buys: the access cannot tear, and the compiler cannot elide, duplicate, or
 * invent it. That is the whole list.
 *
 * It does NOT order these two flags against each other, or against bleMtu. Two
 * relaxed accesses may be observed in either order by the other core. So
 * reading bleMtuOk and then bleMtu as a PAIR is still unsafe -- nothing here
 * says bleMtu was published before the flag that advertises it. The reason that
 * does not bite today is Issue A: notifyChunked() latches bleMtu into a local
 * once, on entry, and range-checks that local against MIN_USABLE_MTU rather
 * than trusting bleMtuOk to imply anything about it. The latch is what removes
 * the need for the pairing, not the atomics. If that latch is ever removed,
 * relaxed is no longer enough and these need acquire/release.
 *
 * bleMtu and bleConnectedAtUs stay volatile deliberately: bleMtu is only ever
 * read through the latch, and bleConnectedAtUs is written and read on core 0
 * alone.
 */
static std::atomic<bool>   bleConnected      {false};
static std::atomic<bool>   bleMtuOk          {false};
static volatile uint16_t   bleMtu            = 0;
static volatile int64_t    bleConnectedAtUs  = 0;
static volatile uint32_t   bleDropped        = 0;  // queue-full drops
static volatile uint32_t   bleSkippedNoConn  = 0;  // frames not offered (no link)
static volatile uint32_t   bleNotifies       = 0;  // notify() calls issued
static volatile uint32_t   bleRefused        = 0;  // size/MTU invariant failed
// ===== END BLE ADDED =====

/* ---------------------------------------------------------------------
 * Sampling + framing
 * ------------------------------------------------------------------- */
#if USE_LOCAL_SAMPLE_HELPERS

static uint16_t sampleChannel(uint8_t pin) {
  if (USE_STUB) {
    // deterministic stub: slow triangle so the host has something to validate
    static uint32_t t = 0;
    t++;
    return (uint16_t)((t + pin * 300) % 4096);
  }
  uint32_t acc = 0;
  for (int i = 0; i < OVERSAMPLE; i++) acc += analogRead(pin);
  return (uint16_t)(acc / OVERSAMPLE);
}

static int buildFrameLine(char* out, size_t n, uint16_t seq, int64_t ts_us,
                          const uint16_t v[6]) {
  // Checksum over the EIGHT integer fields: seq, ts_us, s0..s5.
  // ts_us is summed at full 64-bit width. (The old firmware truncated it to
  // 32 bits, which silently diverges from the Python validator after ~71.5
  // minutes of continuous capture. Identical below that, so this is a safe
  // fix to take now.)
  uint64_t sum = (uint64_t)seq + (uint64_t)ts_us;
  for (int i = 0; i < 6; i++) sum += v[i];
  uint8_t ck = (uint8_t)(sum & 0xFF);

  int want = snprintf(out, n, "INS,%u,%lld,%u,%u,%u,%u,%u,%u,%u\n",
                      (unsigned)seq, (long long)ts_us,
                      v[0], v[1], v[2], v[3], v[4], v[5], (unsigned)ck);

  // snprintf() returns the length it WANTED to write, not the length it wrote.
  // Handing that straight to Serial.write() walks off the end of a 72-byte
  // stack buffer and puts whatever follows it on the wire. Clamp to what is
  // actually in the buffer -- snprintf NUL-terminates at out[n-1], so n-1 bytes
  // of payload landed -- and count it: a truncation here means the frame format
  // grew without MAX_LINE_LEN growing with it, which is a build-time error
  // wearing a runtime disguise.
  if (want < 0)          { frameTruncs++; return 0; }
  if ((size_t)want >= n) { frameTruncs++; return (int)(n - 1); }
  return want;
}
#endif

/* ---------------------------------------------------------------------
 * ===== BLE ADDED =====  BLE plumbing
 * ------------------------------------------------------------------- */
#if BLE_ENABLED

class InsoleServerCB : public BLEServerCallbacks {
  void onConnect(BLEServer* s) override {
    bleMtu           = 0;
    bleMtuOk.store(false, std::memory_order_relaxed);
    bleConnectedAtUs = esp_timer_get_time();
    bleConnected.store(true, std::memory_order_relaxed);
  }
  void onDisconnect(BLEServer* s) override {
    bleConnected.store(false, std::memory_order_relaxed);
    bleMtuOk.store(false, std::memory_order_relaxed);
    bleMtu       = 0;
    // Throw away anything still queued. On reconnect we must not dump a burst
    // of stale frames — that looks like a timing anomaly to the host.
    if (bleQueue) xQueueReset(bleQueue);
    s->startAdvertising();   // become findable again immediately
  }
};

static void bleInit() {
  BLEDevice::init(BLE_DEVICE_NAME);
  BLEDevice::setMTU(517);          // our preferred MTU; the central decides

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new InsoleServerCB());

  BLEService* svc = pServer->createService(NUS_SERVICE_UUID);

  pTxChar = svc->createCharacteristic(NUS_TX_UUID,
                                      BLECharacteristic::PROPERTY_NOTIFY);
  pTxChar->addDescriptor(new BLE2902());   // CCCD — delete with the include if needed

  // RX exists only so the service is a real NUS. We never read it.
  svc->createCharacteristic(NUS_RX_UUID,
                            BLECharacteristic::PROPERTY_WRITE |
                            BLECharacteristic::PROPERTY_WRITE_NR);

  svc->start();

  BLEAdvertising* adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE_UUID);   // so nRF Connect shows "Nordic UART Service"
  adv->setScanResponse(true);
  // Preferred connection interval, in 1.25 ms units: 0x06 = 7.5 ms (the BLE
  // floor), 0x12 = 22.5 ms. The second call used to be setMinPreferred as well,
  // so it overwrote the first and the range was never expressed. At 100 Hz with
  // FRAMES_PER_NOTIFY=3 we need ~33 notifications/s, one every ~30 ms, so the
  // 22.5 ms ceiling is the whole point of the hint. It stays a hint: the
  // central picks the actual interval.
  adv->setMinPreferred(0x06);
  adv->setMaxPreferred(0x12);
  BLEDevice::startAdvertising();
}

/* Core-0 diagnostics go through here. Core 0 never touches Serial directly.
 *
 * Two cores writing one USB CDC peripheral with no mutual exclusion can
 * interleave mid-line, and the victim is a *serial* data frame -- so it
 * surfaces on the host as a random checksum failure during a walking trial,
 * pointing nowhere near the logging that caused it. Serial is therefore owned
 * by core 1: core 0 formats into a queue message and moves on.
 *
 * A mutex was the alternative and was rejected. Serial.write() on native USB
 * CDC blocks when the host is not draining, so a mutex held by core 0 could be
 * waited on by core 1 inside the frame-emit path -- exactly the coupling the
 * core split exists to prevent.
 *
 * Zero-timeout send, drop on full: a diagnostic is never worth stalling the
 * BLE consumer for. */
static void bleLog(const char* fmt, ...) {
  if (!logQueue) return;

  LineMsg m;
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(m.text, MAX_LINE_LEN, fmt, ap);
  va_end(ap);

  if (n < 0) return;
  if (n >= MAX_LINE_LEN) n = MAX_LINE_LEN - 1;   // same clamp rule as above
  m.len = (uint8_t)n;

  xQueueSend(logQueue, &m, 0);   // never blocks; a lost log line is acceptable
}

/* Send a buffer, splitting on NEWLINE boundaries so a CSV line is never cut
 * across two notifications by us. (The host still has to reassemble, because
 * the *link layer* can fragment, but we never manufacture a split mid-field.)
 *
 * Returns false if nothing was sent because an invariant did not hold. The
 * caller owns the accounting, because only it knows how many frames the buffer
 * held.
 *
 * Two changes make the mid-line cut unreachable by construction rather than by
 * luck:
 *
 *   1. bleMtu is latched into a local ONCE, here. onDisconnect() zeroes it from
 *      the callback context, so re-reading it inside the walk could drop
 *      `usable` to the old 20-byte floor partway through a buffer already sized
 *      against a negotiated 100+ byte MTU -- and a 20-byte window is narrower
 *      than a frame, so every chunk after that point would be a fragment.
 *
 *   2. MIN_USABLE_MTU (100) - 3 = 97 usable bytes, and bleEnqueue() refuses any
 *      line >= MAX_LINE_LEN (72). A whole line therefore always fits in one
 *      chunk, so a window that starts on a line boundary always contains a
 *      newline. If one somehow does not, the frame format has outgrown the MTU
 *      floor: that is a configuration error, so refuse the batch and count it
 *      rather than putting a fragment on the wire for the host to splice into a
 *      syntactically valid frame carrying wrong values. */
static bool notifyChunked(const char* buf, size_t len) {
  const uint16_t mtu = bleMtu;               // latch once -- see (1)
  if (mtu < MIN_USABLE_MTU) return false;    // disconnected under us

  const size_t usable = (size_t)(mtu - 3);

  size_t off = 0;
  while (off < len && bleConnected.load(std::memory_order_relaxed)) {
    size_t take = len - off;
    if (take > usable) {
      take = usable;
      size_t nl = 0;
      for (size_t i = take; i > 0; --i) {
        if (buf[off + i - 1] == '\n') { nl = i; break; }
      }
      if (!nl) return false;      // see (2): never emit a fragment
      take = nl;                  // back up to the last complete line
    }
    pTxChar->setValue((uint8_t*)(buf + off), take);
    pTxChar->notify();            // MAY BLOCK — that is why we are on core 0
    bleNotifies++;
    off += take;
    if (off < len) vTaskDelay(1); // let the stack breathe between chunks
  }
  return true;
}

/* The BLE consumer. Pinned to core 0. The Arduino loop() (the sampler) runs on
 * core 1. A blocking notify() therefore CANNOT delay the sample clock — they
 * are different tasks on different cores, communicating only through a queue
 * that the producer writes with a zero timeout. That is the structural
 * guarantee, not a "should be fast enough" argument. */
static void bleTask(void* arg) {
  char    batch[MAX_LINE_LEN * FRAMES_PER_NOTIFY + 8];
  LineMsg msg;

  for (;;) {
    // --- MTU gate: negotiation completes shortly AFTER onConnect fires, so
    //     poll rather than read it once in the callback.
    if (bleConnected.load(std::memory_order_relaxed) &&
        !bleMtuOk.load(std::memory_order_relaxed)) {
      uint16_t m = pServer->getPeerMTU(pServer->getConnId());
      if (m >= MIN_USABLE_MTU) {
        bleMtu   = m;
        bleMtuOk.store(true, std::memory_order_relaxed);
        bleLog("# ble mtu=%u ok\n", (unsigned)m);
      } else if (esp_timer_get_time() - bleConnectedAtUs > MTU_WAIT_US) {
        bleLog("# ble mtu=%u BELOW %d - refusing to notify\n",
               (unsigned)m, MIN_USABLE_MTU);
        bleConnectedAtUs = esp_timer_get_time();   // re-log in 2 s, keep refusing
      }
    }

    // --- gather up to FRAMES_PER_NOTIFY whole lines
    size_t used = 0;
    int    have = 0;
    while (have < FRAMES_PER_NOTIFY) {
      if (xQueueReceive(bleQueue, &msg, pdMS_TO_TICKS(50)) != pdTRUE) break;
      if (used + msg.len >= sizeof(batch)) {
        // Unreachable at the current FRAMES_PER_NOTIFY and frame width
        // (3 x ~57 = 171 < 224), but it stops being unreachable the moment
        // either one changes, and the failure mode is invisible loss: the
        // message was already taken off the queue, so not counting it here
        // loses a frame that no counter ever mentions. Count it and stop
        // gathering -- the next frame is the same size and will not fit
        // either. No logging inside the loop: that would turn a loss event
        // into a timing problem.
        bleRefused++;
        break;
      }
      memcpy(batch + used, msg.text, msg.len);
      used += msg.len;
      have++;
    }
    if (used == 0) continue;

    // Not connected, or MTU too small -> the batch is dropped on the floor.
    if (!bleConnected.load(std::memory_order_relaxed) ||
        !bleMtuOk.load(std::memory_order_relaxed)) { bleDropped += have; continue; }

    // A refusal after some chunks already went out over-counts, but that path
    // is the can't-happen one in (2) above; over-counting a can't-happen is the
    // right way round.
    if (!notifyChunked(batch, used)) bleRefused += have;
  }
}

/* Producer side, called from loop(). NEVER blocks: zero-tick queue send.
 * Drop policy on a full queue is DROP OLDEST — the newest frames are the ones
 * worth keeping, and dropping the head keeps latency bounded. */
static inline void bleEnqueue(const char* line, size_t len) {
  // Relaxed: a plain 32-bit load, no barrier. This line is on the 100 Hz path.
  if (!bleConnected.load(std::memory_order_relaxed))
                          { bleSkippedNoConn++; return; }
  if (len >= MAX_LINE_LEN) { bleDropped++;      return; }

  LineMsg m;
  m.len = (uint8_t)len;
  memcpy(m.text, line, len);

  if (xQueueSend(bleQueue, &m, 0) != pdTRUE) {
    LineMsg junk;
    if (xQueueReceive(bleQueue, &junk, 0) == pdTRUE) bleDropped++;  // evict oldest
    if (xQueueSend(bleQueue, &m, 0) != pdTRUE)       bleDropped++;  // still full: drop new
  }
}
#endif  // BLE_ENABLED
// ===== END BLE ADDED =====

/* ---------------------------------------------------------------------
 * setup / loop
 * ------------------------------------------------------------------- */
static uint16_t seq        = 0;
static int64_t  nextDueUs  = 0;
static int64_t  lastStatUs = 0;

void setup() {
  Serial.begin(921600);          // baud is ignored on native USB CDC
  delay(300);

  analogReadResolution(12);
  for (int i = 0; i < 6; i++) {
    // ADC_11db on core 2.x; core 3.x accepts it (aliased to ADC_ATTEN_DB_12).
    analogSetPinAttenuation(PINS[i], ADC_11db);
  }

  // ===== BLE ADDED =====
#if BLE_ENABLED
  bleQueue = xQueueCreate(BLE_QUEUE_LEN, sizeof(LineMsg));
  // Created before bleInit() so a connect callback firing early has somewhere
  // to put its diagnostics. bleLog() no-ops on a null handle regardless.
  logQueue = xQueueCreate(LOG_QUEUE_LEN, sizeof(LineMsg));
  bleInit();
  // Core 0 = the core the BLE stack already lives on. Arduino loop() is core 1.
  xTaskCreatePinnedToCore(bleTask, "bleTx", 4096, nullptr, 1, nullptr, 0);
  Serial.println("# ble advertising as " BLE_DEVICE_NAME);
#endif
  // ===== END BLE ADDED =====

  nextDueUs  = esp_timer_get_time();
  lastStatUs = nextDueUs;
}

void loop() {
  int64_t now = esp_timer_get_time();
  if (now < nextDueUs) {
    int64_t slack = nextDueUs - now;
    if (slack > 1500) vTaskDelay(pdMS_TO_TICKS(1));   // yield, keep the WDT happy
    return;
  }

  // Absolute-deadline advance: the deadline is derived from the schedule, not
  // from "now", so per-iteration jitter does not accumulate into drift.
  nextDueUs += PERIOD_US;
  if (esp_timer_get_time() - nextDueUs > 5 * PERIOD_US) {
    nextDueUs = esp_timer_get_time();   // we fell far behind; resync, don't spin
  }

  uint16_t v[6];
  for (int i = 0; i < 6; i++) v[i] = sampleChannel(PINS[i]);

  char line[MAX_LINE_LEN];
  int  len = buildFrameLine(line, sizeof(line), seq, now, v);
  seq++;

  Serial.write((const uint8_t*)line, len);   // primary path, unconditional

  // ===== BLE ADDED =====
#if BLE_ENABLED
  bleEnqueue(line, (size_t)len);             // best-effort, non-blocking

  if (now - lastStatUs >= 1000000) {
    lastStatUs = now;

    // Core 0 posts its diagnostics to logQueue rather than writing Serial
    // itself, so only this core ever touches the port and two writes cannot
    // interleave mid-line. Drained HERE, inside the block that already emits a
    // diagnostic once a second, rather than per iteration: the 100 Hz path
    // gains nothing at all, not even a queue poll. Bounded work -- at most
    // LOG_QUEUE_LEN short lines, and the two log sites fire at most once
    // per 2 s.
    if (logQueue) {
      LineMsg lm;
      for (int i = 0; i < LOG_QUEUE_LEN &&
                      xQueueReceive(logQueue, &lm, 0) == pdTRUE; i++) {
        Serial.write((const uint8_t*)lm.text, lm.len);
      }
    }

    // '#' prefix so the host validator can skip it instead of counting it
    // as a malformed frame.
    // refused = size/MTU invariant failures (never expected, unlike drop).
    // trunc   = buildFrameLine() hit MAX_LINE_LEN (never expected either).
    Serial.printf("# ble conn=%d mtu=%u notif=%lu drop=%lu noconn=%lu "
                  "refused=%lu trunc=%lu\n",
                  bleConnected.load(std::memory_order_relaxed) ? 1 : 0, (unsigned)bleMtu,
                  (unsigned long)bleNotifies,
                  (unsigned long)bleDropped,
                  (unsigned long)bleSkippedNoConn,
                  (unsigned long)bleRefused,
                  (unsigned long)frameTruncs);
  }
#endif
  // ===== END BLE ADDED =====
}
