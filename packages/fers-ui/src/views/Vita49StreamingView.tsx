// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).

import FilterListIcon from '@mui/icons-material/FilterList';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import {
    Alert,
    Box,
    Button,
    Chip,
    FormControl,
    Grid,
    InputLabel,
    MenuItem,
    Paper,
    Select,
    Stack,
    Switch,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from '@mui/material';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { dirname } from '@tauri-apps/api/path';
import React, { useEffect, useMemo, useState } from 'react';
import { useScenarioStore } from '@/stores/scenarioStore';
import { getBlockingFmcwValidationMessage } from '@/stores/scenarioStore/fmcwValidation';
import {
    normalizeSimulationOutputMetadata,
    type RawSimulationOutputMetadata,
} from '@/stores/simulationProgressStore';
import {
    deriveExpectedVita49Streams,
    mergeVita49StreamRows,
    toVita49BackendConfig,
    useVita49StreamingStore,
    type Vita49PacketTraceEvent,
    type Vita49StreamStatsEvent,
    validateVita49Config,
} from '@/stores/vita49StreamingStore';

const parsePayload = <T,>(payload: T | string): T =>
    typeof payload === 'string' ? (JSON.parse(payload) as T) : payload;

const formatMetric = (value: number | null | undefined) =>
    value === null || value === undefined
        ? '-'
        : value.toLocaleString(undefined, { maximumSignificantDigits: 6 });

const formatStreamId = (streamId: number | null | undefined) =>
    streamId === null || streamId === undefined
        ? '-'
        : `0x${streamId.toString(16).toUpperCase().padStart(8, '0')}`;

export const Vita49StreamingView = React.memo(function Vita49StreamingView() {
    const config = useVita49StreamingStore((state) => state.config);
    const runState = useVita49StreamingStore((state) => state.runState);
    const expectedStreams = useVita49StreamingStore(
        (state) => state.expectedStreams
    );
    const streamStats = useVita49StreamingStore((state) => state.streamStats);
    const packetTrace = useVita49StreamingStore((state) => state.packetTrace);
    const omittedPacketTraceEvents = useVita49StreamingStore(
        (state) => state.omittedPacketTraceEvents
    );
    const finalVita49Metadata = useVita49StreamingStore(
        (state) => state.finalVita49Metadata
    );
    const error = useVita49StreamingStore((state) => state.error);
    const setConfig = useVita49StreamingStore((state) => state.setConfig);
    const startRun = useVita49StreamingStore((state) => state.startRun);
    const markStopping = useVita49StreamingStore((state) => state.markStopping);
    const setStreamStats = useVita49StreamingStore(
        (state) => state.setStreamStats
    );
    const appendPacketBatch = useVita49StreamingStore(
        (state) => state.appendPacketBatch
    );
    const completeRun = useVita49StreamingStore((state) => state.completeRun);
    const cancelRun = useVita49StreamingStore((state) => state.cancelRun);
    const failRun = useVita49StreamingStore((state) => state.failRun);
    const showError = useScenarioStore((state) => state.showError);
    const showWarning = useScenarioStore((state) => state.showWarning);
    const showSuccess = useScenarioStore((state) => state.showSuccess);
    const scenarioFilePath = useScenarioStore(
        (state) => state.scenarioFilePath
    );
    const outputDirectory = useScenarioStore((state) => state.outputDirectory);

    const [metadataExportPath, setMetadataExportPath] = useState<string | null>(
        null
    );
    const [streamFilter, setStreamFilter] = useState('all');
    const [packetKindFilter, setPacketKindFilter] = useState('all');
    const [droppedOnly, setDroppedOnly] = useState(false);
    const [overRangeOnly, setOverRangeOnly] = useState(false);
    const [sampleLossOnly, setSampleLossOnly] = useState(false);

    useEffect(() => {
        let active = true;
        const unlisteners = Promise.all([
            listen<string>('vita49-stream-stats', (event) => {
                if (!active) return;
                setStreamStats(
                    parsePayload<Vita49StreamStatsEvent>(event.payload)
                );
            }),
            listen<string>('vita49-packet-batch', (event) => {
                if (!active) return;
                appendPacketBatch(
                    parsePayload<Vita49PacketTraceEvent[]>(event.payload)
                );
            }),
            listen<string>('vita49-output-metadata', (event) => {
                if (!active) return;
                const metadata = normalizeSimulationOutputMetadata(
                    JSON.parse(event.payload) as RawSimulationOutputMetadata
                );
                completeRun(metadata);
            }),
            listen<string>('vita49-stream-complete', (event) => {
                if (!active) return;
                const metadata = normalizeSimulationOutputMetadata(
                    JSON.parse(event.payload) as RawSimulationOutputMetadata
                );
                completeRun(metadata);
            }),
            listen<string>('vita49-stream-cancelled', (event) => {
                if (!active) return;
                const metadata = normalizeSimulationOutputMetadata(
                    JSON.parse(event.payload) as RawSimulationOutputMetadata
                );
                cancelRun(metadata);
            }),
            listen<string>('vita49-stream-error', (event) => {
                if (!active) return;
                failRun(event.payload);
                showError(`VITA49 streaming failed: ${event.payload}`);
            }),
        ]);

        return () => {
            active = false;
            unlisteners.then((listeners) =>
                listeners.forEach((unlisten) => unlisten())
            );
        };
    }, [
        appendPacketBatch,
        cancelRun,
        completeRun,
        failRun,
        setStreamStats,
        showError,
    ]);

    const streamRows = useMemo(
        () => mergeVita49StreamRows(expectedStreams, streamStats),
        [expectedStreams, streamStats]
    );

    const aggregate = useMemo(
        () =>
            streamRows.reduce(
                (acc, row) => ({
                    packets: acc.packets + row.packetsEmitted,
                    samples: acc.samples + row.samplesEmitted,
                    drops: acc.drops + row.packetsDropped,
                    late: acc.late + row.latePacketCount,
                    overRange: acc.overRange + row.overRangeCount,
                    context: acc.context + row.contextPackets,
                }),
                {
                    packets: 0,
                    samples: 0,
                    drops: 0,
                    late: 0,
                    overRange: 0,
                    context: 0,
                }
            ),
        [streamRows]
    );

    const streamIdOptions = useMemo(() => {
        const ids = new Set<number>();
        for (const row of streamRows) {
            if (row.streamId !== null) ids.add(row.streamId);
        }
        for (const packet of packetTrace) {
            if (packet.stream_id) ids.add(packet.stream_id);
        }
        return Array.from(ids).sort((a, b) => a - b);
    }, [packetTrace, streamRows]);

    const filteredPackets = useMemo(
        () =>
            packetTrace.filter((packet) => {
                if (
                    streamFilter !== 'all' &&
                    packet.stream_id !== Number(streamFilter)
                ) {
                    return false;
                }
                if (packetKindFilter === 'data' && !packet.data_packet) {
                    return false;
                }
                if (packetKindFilter === 'context' && !packet.context_packet) {
                    return false;
                }
                if (droppedOnly && !packet.dropped) return false;
                if (overRangeOnly && !packet.over_range) return false;
                if (sampleLossOnly && !packet.sample_loss) return false;
                return true;
            }),
        [
            droppedOnly,
            overRangeOnly,
            packetKindFilter,
            packetTrace,
            sampleLossOnly,
            streamFilter,
        ]
    );

    const isRunning = runState === 'running' || runState === 'stopping';
    const configErrors = validateVita49Config(config);

    const getEffectiveOutputDir = async () => {
        if (outputDirectory) return outputDirectory;
        if (scenarioFilePath) {
            return dirname(scenarioFilePath);
        }
        return '.';
    };

    const handleStart = async () => {
        const scenarioState = useScenarioStore.getState();
        const validationMessage =
            getBlockingFmcwValidationMessage(scenarioState);
        if (validationMessage) {
            showError(`FMCW validation failed: ${validationMessage}`);
            return;
        }

        const errors = validateVita49Config(config);
        if (errors.length > 0) {
            showError(errors.join(' '));
            return;
        }

        const expected = deriveExpectedVita49Streams(scenarioState);
        if (expected.length === 0) {
            showWarning('No CW/FMCW receiver streams are configured.');
        }

        setMetadataExportPath(null);
        startRun(expected);
        try {
            await invoke('set_output_directory', {
                dir: await getEffectiveOutputDir(),
            });
            await scenarioState.syncBackend();
            await invoke('start_vita49_stream', {
                config: toVita49BackendConfig(config),
            });
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            failRun(message);
            showError(`Failed to start VITA49 streaming: ${message}`);
        }
    };

    const handleStop = async () => {
        markStopping();
        try {
            await invoke('stop_simulation');
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            showError(`Failed to stop simulation: ${message}`);
        }
    };

    const exportMetadataJson = async () => {
        try {
            const outputPath = await invoke<string>(
                'export_output_metadata_json'
            );
            setMetadataExportPath(outputPath);
            showSuccess(`Metadata JSON saved to ${outputPath}`);
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            showError(`Failed to export metadata JSON: ${message}`);
        }
    };

    return (
        <Box sx={{ p: 3, height: '100%', overflowY: 'auto' }}>
            <Stack
                direction="row"
                spacing={2}
                alignItems="center"
                sx={{ mb: 2 }}
            >
                <Typography variant="h4">VITA49 Streams</Typography>
                <Chip
                    label={runState}
                    color={isRunning ? 'primary' : 'default'}
                />
            </Stack>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid size={{ xs: 12, lg: 4 }}>
                    <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
                        <Typography variant="h6" sx={{ mb: 2 }}>
                            Runtime
                        </Typography>
                        <Stack spacing={2}>
                            <TextField
                                label="Host"
                                size="small"
                                value={config.host}
                                disabled={isRunning}
                                onChange={(event) =>
                                    setConfig({ host: event.target.value })
                                }
                            />
                            <TextField
                                label="Port"
                                size="small"
                                type="number"
                                value={config.port}
                                disabled={isRunning}
                                onChange={(event) =>
                                    setConfig({
                                        port: Number(event.target.value),
                                    })
                                }
                            />
                            <TextField
                                label="Full-scale"
                                size="small"
                                type="number"
                                value={config.fullscale}
                                disabled={isRunning}
                                onChange={(event) =>
                                    setConfig({
                                        fullscale: Number(event.target.value),
                                    })
                                }
                            />
                            <FormControl size="small">
                                <InputLabel id="vita49-epoch-mode-label">
                                    Epoch
                                </InputLabel>
                                <Select
                                    labelId="vita49-epoch-mode-label"
                                    label="Epoch"
                                    value={config.epochMode}
                                    disabled={isRunning}
                                    onChange={(event) =>
                                        setConfig({
                                            epochMode: event.target
                                                .value as typeof config.epochMode,
                                        })
                                    }
                                >
                                    <MenuItem value="auto">Auto</MenuItem>
                                    <MenuItem value="fixed">Fixed</MenuItem>
                                </Select>
                            </FormControl>
                            {config.epochMode === 'fixed' && (
                                <TextField
                                    label="Unix ns"
                                    size="small"
                                    value={config.epochUnixNanoseconds}
                                    disabled={isRunning}
                                    onChange={(event) =>
                                        setConfig({
                                            epochUnixNanoseconds:
                                                event.target.value,
                                        })
                                    }
                                />
                            )}
                            <TextField
                                label="Max UDP payload"
                                size="small"
                                type="number"
                                value={config.maxUdpPayload}
                                disabled={isRunning}
                                onChange={(event) =>
                                    setConfig({
                                        maxUdpPayload: Number(
                                            event.target.value
                                        ),
                                    })
                                }
                            />
                            <TextField
                                label="Queue depth"
                                size="small"
                                type="number"
                                value={config.queueDepth}
                                disabled={isRunning}
                                onChange={(event) =>
                                    setConfig({
                                        queueDepth: Number(event.target.value),
                                    })
                                }
                            />
                            <TextField
                                label="Trace ring"
                                size="small"
                                type="number"
                                value={config.packetTraceRingSize}
                                disabled={isRunning}
                                onChange={(event) =>
                                    setConfig({
                                        packetTraceRingSize: Number(
                                            event.target.value
                                        ),
                                    })
                                }
                            />
                            {configErrors.length > 0 && (
                                <Alert severity="warning">
                                    {configErrors.join(' ')}
                                </Alert>
                            )}
                            <Stack direction="row" spacing={1}>
                                <Button
                                    variant="contained"
                                    startIcon={<PlayCircleOutlineIcon />}
                                    disabled={
                                        isRunning || configErrors.length > 0
                                    }
                                    onClick={handleStart}
                                >
                                    Start
                                </Button>
                                <Button
                                    variant="outlined"
                                    color="error"
                                    startIcon={<StopCircleIcon />}
                                    disabled={runState !== 'running'}
                                    onClick={handleStop}
                                >
                                    Stop
                                </Button>
                            </Stack>
                        </Stack>
                    </Paper>
                </Grid>

                <Grid size={{ xs: 12, lg: 8 }}>
                    <Stack spacing={2}>
                        <Paper variant="outlined" sx={{ p: 2 }}>
                            <Stack
                                direction={{ xs: 'column', md: 'row' }}
                                spacing={2}
                                justifyContent="space-between"
                            >
                                <Box>
                                    <Typography variant="overline">
                                        Endpoint
                                    </Typography>
                                    <Typography variant="h6">
                                        {config.host}:{config.port}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="overline">
                                        Profile
                                    </Typography>
                                    <Typography variant="h6">
                                        {finalVita49Metadata?.class_id ??
                                            '0xFA52530001000101'}
                                    </Typography>
                                </Box>
                                <Box>
                                    <Typography variant="overline">
                                        Epoch
                                    </Typography>
                                    <Typography variant="h6">
                                        {formatMetric(
                                            streamStats?.epoch_unix_nanoseconds ??
                                                finalVita49Metadata?.epoch_unix_nanoseconds
                                        )}
                                    </Typography>
                                </Box>
                            </Stack>
                        </Paper>

                        <Grid container spacing={1}>
                            {[
                                ['Packets', aggregate.packets],
                                ['Samples', aggregate.samples],
                                ['Drops', aggregate.drops],
                                ['Late', aggregate.late],
                                ['Over-range', aggregate.overRange],
                                ['Context', aggregate.context],
                            ].map(([label, value]) => (
                                <Grid size={{ xs: 6, md: 2 }} key={label}>
                                    <Paper variant="outlined" sx={{ p: 1.5 }}>
                                        <Typography variant="caption">
                                            {label}
                                        </Typography>
                                        <Typography variant="h6">
                                            {formatMetric(value as number)}
                                        </Typography>
                                    </Paper>
                                </Grid>
                            ))}
                        </Grid>
                    </Stack>
                </Grid>
            </Grid>

            <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                <Stack
                    direction="row"
                    justifyContent="space-between"
                    alignItems="center"
                    sx={{ mb: 1 }}
                >
                    <Typography variant="h6">Streams</Typography>
                    <Button
                        variant="outlined"
                        startIcon={<SaveAltIcon />}
                        disabled={!finalVita49Metadata}
                        onClick={exportMetadataJson}
                    >
                        Export JSON
                    </Button>
                </Stack>
                {metadataExportPath && (
                    <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ mb: 1, overflowWrap: 'anywhere' }}
                    >
                        {metadataExportPath}
                    </Typography>
                )}
                <TableContainer>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Receiver</TableCell>
                                <TableCell>Stream ID</TableCell>
                                <TableCell align="right">Rate</TableCell>
                                <TableCell align="right">RF</TableCell>
                                <TableCell align="right">Packets</TableCell>
                                <TableCell align="right">Samples</TableCell>
                                <TableCell align="right">Drops</TableCell>
                                <TableCell align="right">Late</TableCell>
                                <TableCell align="right">Context</TableCell>
                                <TableCell>First</TableCell>
                                <TableCell>Last</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {streamRows.map((row) => (
                                <TableRow key={row.key}>
                                    <TableCell>
                                        <Stack spacing={0.25}>
                                            <Typography variant="body2">
                                                {row.receiverName}
                                            </Typography>
                                            <Typography
                                                variant="caption"
                                                color="text.secondary"
                                            >
                                                {row.platformName || row.mode}
                                            </Typography>
                                        </Stack>
                                    </TableCell>
                                    <TableCell>
                                        {formatStreamId(row.streamId)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(row.sampleRate)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(row.referenceFrequency)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(row.packetsEmitted)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(row.samplesEmitted)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(row.packetsDropped)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(row.latePacketCount)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(row.contextPackets)}
                                    </TableCell>
                                    <TableCell>
                                        {formatMetric(row.firstTimestampUnixPs)}
                                    </TableCell>
                                    <TableCell>
                                        {formatMetric(row.lastTimestampUnixPs)}
                                    </TableCell>
                                </TableRow>
                            ))}
                            {streamRows.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={11}>
                                        <Typography color="text.secondary">
                                            No streams
                                        </Typography>
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Paper>

            <Paper variant="outlined" sx={{ p: 2 }}>
                <Stack
                    direction={{ xs: 'column', md: 'row' }}
                    spacing={2}
                    alignItems={{ xs: 'stretch', md: 'center' }}
                    sx={{ mb: 2 }}
                >
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        Packet Trace
                    </Typography>
                    <FilterListIcon color="action" />
                    <FormControl size="small" sx={{ minWidth: 160 }}>
                        <InputLabel id="vita49-stream-filter-label">
                            Stream
                        </InputLabel>
                        <Select
                            labelId="vita49-stream-filter-label"
                            label="Stream"
                            value={streamFilter}
                            onChange={(event) =>
                                setStreamFilter(event.target.value)
                            }
                        >
                            <MenuItem value="all">All</MenuItem>
                            {streamIdOptions.map((streamId) => (
                                <MenuItem
                                    value={String(streamId)}
                                    key={streamId}
                                >
                                    {formatStreamId(streamId)}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 140 }}>
                        <InputLabel id="vita49-kind-filter-label">
                            Kind
                        </InputLabel>
                        <Select
                            labelId="vita49-kind-filter-label"
                            label="Kind"
                            value={packetKindFilter}
                            onChange={(event) =>
                                setPacketKindFilter(event.target.value)
                            }
                        >
                            <MenuItem value="all">All</MenuItem>
                            <MenuItem value="data">Data</MenuItem>
                            <MenuItem value="context">Context</MenuItem>
                        </Select>
                    </FormControl>
                    {[
                        ['Dropped', droppedOnly, setDroppedOnly],
                        ['Over-range', overRangeOnly, setOverRangeOnly],
                        ['Sample-loss', sampleLossOnly, setSampleLossOnly],
                    ].map(([label, checked, setter]) => (
                        <Stack
                            direction="row"
                            alignItems="center"
                            spacing={0.5}
                            key={label as string}
                        >
                            <Switch
                                size="small"
                                checked={checked as boolean}
                                onChange={(event) =>
                                    (setter as (value: boolean) => void)(
                                        event.target.checked
                                    )
                                }
                            />
                            <Typography variant="body2">
                                {label as string}
                            </Typography>
                        </Stack>
                    ))}
                </Stack>
                {omittedPacketTraceEvents > 0 && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                        {formatMetric(omittedPacketTraceEvents)} packet trace
                        events omitted
                    </Alert>
                )}
                <TableContainer sx={{ maxHeight: 420 }}>
                    <Table size="small" stickyHeader>
                        <TableHead>
                            <TableRow>
                                <TableCell align="right">Seq</TableCell>
                                <TableCell>Event</TableCell>
                                <TableCell>Stream</TableCell>
                                <TableCell align="right">Bytes</TableCell>
                                <TableCell align="right">Samples</TableCell>
                                <TableCell align="right">t</TableCell>
                                <TableCell align="right">Unix ps</TableCell>
                                <TableCell>Flags</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredPackets.map((packet) => (
                                <TableRow key={packet.sequence}>
                                    <TableCell align="right">
                                        {packet.sequence}
                                    </TableCell>
                                    <TableCell>{packet.event}</TableCell>
                                    <TableCell>
                                        {formatStreamId(packet.stream_id)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(packet.byte_count)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(packet.sample_count)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(packet.first_sample_time)}
                                    </TableCell>
                                    <TableCell align="right">
                                        {formatMetric(packet.timestamp_unix_ps)}
                                    </TableCell>
                                    <TableCell>
                                        <Stack direction="row" spacing={0.5}>
                                            {packet.dropped && (
                                                <Chip
                                                    label="drop"
                                                    size="small"
                                                    color="error"
                                                />
                                            )}
                                            {packet.over_range && (
                                                <Chip
                                                    label="over"
                                                    size="small"
                                                    color="warning"
                                                />
                                            )}
                                            {packet.sample_loss && (
                                                <Chip
                                                    label="loss"
                                                    size="small"
                                                    color="warning"
                                                />
                                            )}
                                        </Stack>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {filteredPackets.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={8}>
                                        <Typography color="text.secondary">
                                            No packets
                                        </Typography>
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Paper>
        </Box>
    );
});
