// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2025-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file finalizer.h
 * @brief Declares the functions for the asynchronous receiver finalization pipelines.
 */

#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/receiver_output.h"
#include "core/simulation_state.h"

namespace radar
{
	class Receiver;
	class Target;
}

namespace pool
{
	class ThreadPool;
}

namespace core
{
	struct OutputFileMetadata;
	class OutputMetadataCollector;
	class ProgressReporter;
}

namespace processing
{
	/// Builds the receiver stream descriptor used by output sinks.
	[[nodiscard]] core::ReceiverStreamDescriptor buildReceiverStreamDescriptor(const radar::Receiver* receiver,
																			   RealType sample_rate);

	/// Builds a non-owning output sample block over contiguous processed complex samples.
	[[nodiscard]] core::ReceiverSampleBlock
	buildReceiverSampleBlock(const radar::Receiver* receiver, RealType first_sample_time, RealType sample_rate,
							 std::span<const ComplexType> samples, std::uint64_t sample_start,
							 std::shared_ptr<const core::OutputFileMetadata> file_metadata = nullptr);

	/// HDF5 implementation of the receiver output sink contract.
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

	/// Creates the default HDF5 receiver output sink.
	[[nodiscard]] std::unique_ptr<core::ReceiverOutputSink>
	makeHdf5OutputSink(std::string output_dir,
					   std::shared_ptr<core::OutputMetadataCollector> metadata_collector = nullptr);

	/**
	 * @brief The main function for a dedicated pulsed-mode receiver finalizer thread.
	 *
	 * This function runs in a loop, dequeuing and processing RenderingJobs for a
	 * specific receiver. It handles all expensive rendering, signal processing,
	 * and I/O for that receiver's data.
	 *
	 * @param receiver A pointer to the pulsed-mode receiver to process.
	 * @param targets A pointer to the world's list of targets for interference calculation.
	 * @param reporter Shared pointer to the progress reporter for status updates.
	 * @param output_dir Output directory for the simulation files.
	 */
	void runPulsedFinalizer(radar::Receiver* receiver, const std::vector<std::unique_ptr<radar::Target>>* targets,
							std::shared_ptr<core::ProgressReporter> reporter, const std::string& output_dir,
							std::shared_ptr<core::OutputMetadataCollector> metadata_collector = nullptr,
							core::ReceiverOutputSink* output_sink = nullptr);

	/**
	 * @brief The finalization task for a streaming-mode receiver.
	 *
	 * This function is submitted to the main thread pool when a streaming receiver
	 * finishes its operation. It processes the entire collected I/Q buffer,
	 * applies interference and noise, and writes the final data to a file.
	 *
	 * @param receiver A pointer to the streaming-mode receiver to finalize.
	 * @param pool A pointer to the main thread pool for parallelizing sub-tasks.
	 * @param reporter Shared pointer to the progress reporter for status updates.
	 * @param output_dir Output directory for the simulation files.
	 */
	void finalizeStreamingReceiver(radar::Receiver* receiver, pool::ThreadPool* pool,
								   std::shared_ptr<core::ProgressReporter> reporter, const std::string& output_dir,
								   std::shared_ptr<core::OutputMetadataCollector> metadata_collector = nullptr,
								   std::vector<core::ActiveStreamingSource> streaming_sources = {},
								   core::ReceiverOutputSink* output_sink = nullptr);
}
