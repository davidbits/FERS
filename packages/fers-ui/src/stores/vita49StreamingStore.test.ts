// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).

import { describe, expect, test } from 'bun:test';
import { defaultGlobalParameters } from './scenarioStore/defaults';
import type { ScenarioData } from './scenarioStore/types';
import {
    DEFAULT_VITA49_CONFIG,
    deriveExpectedVita49Streams,
    toVita49BackendConfig,
    useVita49StreamingStore,
    type Vita49PacketTraceEvent,
    validateVita49Config,
} from './vita49StreamingStore';

const packet = (sequence: number): Vita49PacketTraceEvent => ({
    sequence,
    event: 'data',
    stream_id: 10,
    byte_count: 128,
    sample_count: 24,
    first_sample_time: 0,
    timestamp_unix_ps: 0,
    data_packet: true,
    context_packet: false,
    dropped: false,
    over_range: false,
    sample_loss: false,
});

describe('VITA49 streaming config', () => {
    test('validates malformed endpoint and runtime bounds', () => {
        expect(
            validateVita49Config({
                ...DEFAULT_VITA49_CONFIG,
                host: ' ',
                port: 0,
                fullscale: 0,
                maxUdpPayload: 63,
                queueDepth: 0,
                packetTraceRingSize: 0,
            })
        ).toEqual([
            'Host is required.',
            'Port must be an integer from 1 to 65535.',
            'Full-scale must be a positive finite number.',
            'Max UDP payload must be an integer from 64 to 65507.',
            'Queue depth must be a positive integer.',
            'Packet trace ring size must be a positive integer.',
        ]);
    });

    test('validates fixed epoch and preserves ns as a string for Rust', () => {
        expect(
            validateVita49Config({
                ...DEFAULT_VITA49_CONFIG,
                epochMode: 'fixed',
                epochUnixNanoseconds: '4294967296000000000',
            })
        ).toContain('Fixed epoch must fit the VRT 32-bit UTC seconds field.');

        expect(
            toVita49BackendConfig({
                ...DEFAULT_VITA49_CONFIG,
                epochMode: 'fixed',
                epochUnixNanoseconds: '1700000000123456789',
            })
        ).toMatchObject({
            epoch_unix_nanoseconds: '1700000000123456789',
            trace_enabled: true,
            packet_trace_ring_size: DEFAULT_VITA49_CONFIG.packetTraceRingSize,
        });
    });
});

describe('VITA49 expected stream derivation', () => {
    test('covers receiver and monostatic streaming components', () => {
        const scenario: Pick<
            ScenarioData,
            'globalParameters' | 'platforms' | 'waveforms'
        > = {
            globalParameters: { ...defaultGlobalParameters, rate: 2e6 },
            waveforms: [
                {
                    id: '100',
                    type: 'Waveform',
                    name: 'CW',
                    waveformType: 'cw',
                    power: 1,
                    carrier_frequency: 9.6e9,
                },
            ],
            platforms: [
                {
                    id: '1',
                    type: 'Platform',
                    name: 'Platform A',
                    motionPath: {
                        interpolation: 'static',
                        waypoints: [
                            { id: '1', x: 0, y: 0, altitude: 0, time: 0 },
                        ],
                    },
                    rotation: {
                        type: 'fixed',
                        startAzimuth: 0,
                        startElevation: 0,
                        azimuthRate: 0,
                        elevationRate: 0,
                    },
                    components: [
                        {
                            id: '10',
                            type: 'receiver',
                            name: 'Rx',
                            radarType: 'cw',
                            window_skip: null,
                            window_length: null,
                            prf: null,
                            antennaId: null,
                            timingId: null,
                            noiseTemperature: null,
                            noDirectPaths: false,
                            noPropagationLoss: false,
                            schedule: [],
                        },
                        {
                            id: '11',
                            type: 'monostatic',
                            name: 'Mono',
                            txId: '12',
                            rxId: '13',
                            radarType: 'fmcw',
                            window_skip: null,
                            window_length: null,
                            prf: null,
                            antennaId: null,
                            waveformId: '100',
                            timingId: null,
                            noiseTemperature: null,
                            noDirectPaths: false,
                            noPropagationLoss: false,
                            fmcwModeConfig: { if_sample_rate: 250000 },
                            schedule: [],
                        },
                    ],
                },
            ],
        };

        expect(deriveExpectedVita49Streams(scenario)).toMatchObject([
            {
                receiverId: 10,
                receiverName: 'Rx',
                sampleRate: 2e6,
                referenceFrequency: null,
            },
            {
                receiverId: 13,
                receiverName: 'Mono',
                sampleRate: 250000,
                referenceFrequency: 9.6e9,
            },
        ]);
    });
});

describe('VITA49 packet trace ring', () => {
    test('bounds packets and counts omitted events', () => {
        useVita49StreamingStore.setState({
            config: { ...DEFAULT_VITA49_CONFIG, packetTraceRingSize: 3 },
            packetTrace: [],
            omittedPacketTraceEvents: 0,
        });

        useVita49StreamingStore
            .getState()
            .appendPacketBatch([1, 2, 3, 4, 5].map(packet));

        const state = useVita49StreamingStore.getState();
        expect(state.packetTrace.map((entry) => entry.sequence)).toEqual([
            3, 4, 5,
        ]);
        expect(state.omittedPacketTraceEvents).toBe(2);
    });

    test('adds backend omitted packet count when appending polled batches', () => {
        useVita49StreamingStore.setState({
            config: { ...DEFAULT_VITA49_CONFIG, packetTraceRingSize: 5 },
            packetTrace: [],
            omittedPacketTraceEvents: 0,
        });

        useVita49StreamingStore
            .getState()
            .appendPacketBatch([1, 2].map(packet), 7);

        const state = useVita49StreamingStore.getState();
        expect(state.packetTrace.map((entry) => entry.sequence)).toEqual([
            1, 2,
        ]);
        expect(state.omittedPacketTraceEvents).toBe(7);
    });
});
