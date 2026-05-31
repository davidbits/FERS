// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <memory>
#include <string>

#include "core/receiver_output.h"

namespace core
{
	class OutputMetadataCollector;
}

namespace serial
{
	class Hdf5OutputSink final : public core::ReceiverOutputSink
	{
	public:
		Hdf5OutputSink(std::string output_dir,
					   std::shared_ptr<core::OutputMetadataCollector> metadata_collector = nullptr);
		~Hdf5OutputSink() override;

		Hdf5OutputSink(const Hdf5OutputSink&) = delete;
		Hdf5OutputSink& operator=(const Hdf5OutputSink&) = delete;

		void initializeRun(const core::OutputConfig& config, std::string simulation_name) override;
		std::uint32_t registerStream(const core::ReceiverStreamDescriptor& stream) override;
		void openStream(std::uint32_t stream_id, RealType first_sample_time) override;
		void submitBlock(const core::ReceiverSampleBlock& block) override;
		void emitContextHeartbeat(RealType simulation_time) override;
		void closeStream(std::uint32_t stream_id) override;
		core::OutputStats finalize() override;

	private:
		struct Impl;
		std::unique_ptr<Impl> _impl;
	};

	[[nodiscard]] std::unique_ptr<core::ReceiverOutputSink>
	makeHdf5OutputSink(std::string output_dir,
					   std::shared_ptr<core::OutputMetadataCollector> metadata_collector = nullptr);
}
