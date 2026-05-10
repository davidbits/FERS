# FERS VITA 49.2 Streaming Implementation

This page is the implementation contract for the FERS VITA 49.2 UDP output backend.

## Frozen Profile

- Packet family: ANSI/VITA 49.2-2017 (R2024).
- Transport: raw VRT packets over UDP/IP; no VITA 49.1 wrapper.
- Endpoint model: one configured destination `host:port`.
- Stream model: one VRT Stream ID per FERS receiver.
- Payload: complex Cartesian IQ, I then Q, signed 16-bit two's-complement, network byte order.
- Timestamp model: TSI UTC and TSF real-time picoseconds; timestamp is the first sample in the packet.
- Packet sizing: maximum UDP payload is 1400 bytes by default.
- Context cadence: stream open, metadata changes, one-second heartbeat, stream close.
- Trailer indicators: valid data, calibrated time, reference lock, over-range, sample loss.
- Class ID: internal placeholder for `FERS VRT IQ Stream v1` until an official allocation is assigned.
- ADC scaling: fixed full-scale only in VITA mode; HDF5 keeps existing full-buffer normalization.
- Pacing: `std::chrono::steady_clock` scheduler mapped to UTC packet timestamps at run start.
- Backpressure: bounded queue, drop unsent data packets, set sample-loss indicators, continue simulation.

## Runtime Selection

`.fersxml` remains a scenario description and does not select network output. VITA streaming is selected only through the C API or CLI runtime switches.

CLI contract:

```text
--vita49 host:port
--vita49-fullscale <positive-real>
--vita49-epoch <unix-nanoseconds>
```

HDF5 remains the default when `--vita49` is absent.

C API control-plane contract:

```c
int fers_enable_vita49_udp_output(fers_context_t* context, const char* host, uint16_t port);
int fers_set_vita49_fullscale(fers_context_t* context, double fullscale);
int fers_set_vita49_epoch_unix_nanoseconds(fers_context_t* context, uint64_t epoch_unix_nanoseconds);
int fers_set_vita49_max_udp_payload(fers_context_t* context, uint16_t max_udp_payload);
int fers_set_vita49_queue_depth(fers_context_t* context, uint32_t queue_depth);
```

The CLI full-scale switch is required whenever `--vita49` is present. The API validates the same condition before `fers_run_simulation`.

## Metadata Contract

`fers_get_last_output_metadata_json` includes `vita49` only in VITA mode. The section contains endpoint, epoch, class ID, fixed full-scale, maximum UDP payload, queue depth, and a `streams` array. Each stream entry contains receiver ID/name, VRT Stream ID, sample rate, reference frequency, packets/samples emitted, packets/samples dropped, over-range count, late-packet count, context-packet count, and first/last Unix-picosecond timestamps.

Current internal placeholder Class ID: `0xFA52530001000101`. This is not an assigned OUI and must stay documented in the ICD until replaced.

## Shared Interfaces

The cross-module contracts are `core::OutputConfig`, `core::Vita49OutputConfig`, `core::ReceiverOutputSink`, `core::ReceiverStreamDescriptor`, `core::ReceiverSampleBlock`, and `core::OutputStats`.

VITA output is implemented at the receiver-output boundary. `hdf5_handler` must remain unaware of VITA.
