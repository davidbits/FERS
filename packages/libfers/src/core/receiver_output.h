// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/output_config.h"
#include "core/sim_id.h"

namespace core
{
	struct OutputFileMetadata;

	struct ReceiverStreamDescriptor
	{
		SimId receiver_id = 0;
		std::string receiver_name;
		std::string mode;
		RealType sample_rate = 0.0;
		RealType reference_frequency = 0.0;
		RealType if_offset = 0.0;
		RealType bandwidth = 0.0;
		bool dechirped = false;
		bool if_resampled = false;
		unsigned adc_bits = 0;
	};

	struct ReceiverSampleBlock
	{
		ReceiverStreamDescriptor stream;
		RealType first_sample_time = 0.0;
		RealType sample_rate = 0.0;
		std::span<const ComplexType> samples;
		std::uint64_t sample_start = 0;
		bool valid_data = true;
		bool calibrated_time = true;
		bool reference_lock = true;
		std::shared_ptr<const OutputFileMetadata> file_metadata = nullptr;
	};

	struct ReceiverStreamStats
	{
		SimId receiver_id = 0;
		std::string receiver_name;
		std::uint32_t stream_id = 0;
		RealType sample_rate = 0.0;
		RealType reference_frequency = 0.0;
		std::uint64_t packets_emitted = 0;
		std::uint64_t context_packets = 0;
		std::uint64_t samples_emitted = 0;
		std::uint64_t packets_dropped = 0;
		std::uint64_t samples_dropped = 0;
		std::uint64_t over_range_count = 0;
		std::uint64_t late_packet_count = 0;
		std::uint64_t first_timestamp_unix_ps = 0;
		std::uint64_t last_timestamp_unix_ps = 0;
	};

	struct OutputStats
	{
		OutputMode mode = OutputMode::Hdf5;
		std::optional<std::uint64_t> epoch_unix_nanoseconds = std::nullopt;
		std::vector<ReceiverStreamStats> streams;
	};

	class ReceiverOutputSink
	{
	public:
		virtual ~ReceiverOutputSink() = default;

		virtual void initializeRun(const OutputConfig& config, std::string simulation_name) = 0;
		virtual std::uint32_t registerStream(const ReceiverStreamDescriptor& stream) = 0;
		virtual void openStream(std::uint32_t stream_id, RealType first_sample_time) = 0;
		virtual void submitBlock(const ReceiverSampleBlock& block) = 0;
		virtual void emitContextHeartbeat(RealType simulation_time) = 0;
		virtual void closeStream(std::uint32_t stream_id) = 0;
		virtual OutputStats finalize() = 0;
	};
}
