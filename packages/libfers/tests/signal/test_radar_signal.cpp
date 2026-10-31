#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <optional>
#include <span>
#include <vector>

#include "core/parameters.h"
#include "core/sim_id.h"
#include "interpolation/interpolation_filter.h"
#include "interpolation/interpolation_point.h"
#include "signal/radar_signal.h"

using Catch::Matchers::WithinAbs;

namespace
{
	struct ParamGuard
	{
		params::Parameters saved;
		ParamGuard() : saved(params::params) {}
		ParamGuard(const ParamGuard&) = delete;
		ParamGuard& operator=(const ParamGuard&) = delete;
		ParamGuard(ParamGuard&&) = delete;
		ParamGuard& operator=(ParamGuard&&) = delete;
		~ParamGuard() { params::params = saved; }
	};

	ComplexType expectedConstantRender(const std::span<const RealType> filter, const RealType amplitude,
									   const RealType phase)
	{
		const int filt_length = static_cast<int>(filter.size());
		RealType sum = 0.0;
		for (int j = -filt_length / 2; j < filt_length / 2; ++j)
		{
			const int idx = j + filt_length / 2;
			if (idx >= 0 && idx < filt_length)
			{
				sum += filter[static_cast<size_t>(idx)];
			}
		}
		return amplitude * std::exp(ComplexType{0.0, 1.0} * phase) * sum;
	}
}

TEST_CASE("FileWaveform preserves CW/FMCW classification", "[signal][radar][file]")
{
	fers_signal::FileWaveform cw_wave{};
	cw_wave.setKind(fers_signal::FileWaveformKind::Cw);
	cw_wave.load(std::vector<ComplexType>{{1.0, 0.0}}, 1, 8.0);
	fers_signal::RadarSignal const cw("cw-file", 1.0, 10.0, std::move(cw_wave), 1);

	fers_signal::FileWaveform fmcw_wave{};
	fmcw_wave.setKind(fers_signal::FileWaveformKind::Fmcw);
	fmcw_wave.load(std::vector<ComplexType>{{1.0, 0.0}}, 1, 8.0);
	fers_signal::RadarSignal const fmcw("fmcw-file", 1.0, 10.0, std::move(fmcw_wave), 2);

	REQUIRE(cw.isCw());
	REQUIRE(cw.isFileCw());
	REQUIRE_FALSE(cw.isFmcwFamily());
	REQUIRE(cw.getFileWaveform()->getKind() == fers_signal::FileWaveformKind::Cw);
	REQUIRE_FALSE(fmcw.isCw());
	REQUIRE(fmcw.isFileFmcw());
	REQUIRE(fmcw.isFmcwFamily());
	REQUIRE(fmcw.getFileWaveform()->getKind() == fers_signal::FileWaveformKind::Fmcw);
}

TEST_CASE("FileWaveform interpolates a finite complex envelope without wrapping", "[signal][radar][file][hdf5]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	constexpr RealType rate = 32.0;
	constexpr unsigned sample_count = 64;
	std::vector<ComplexType> samples(sample_count);
	for (unsigned i = 0; i < sample_count; ++i)
	{
		const RealType phase = 2.0 * PI * 3.0 * static_cast<RealType>(i) / rate;
		samples[i] = {std::cos(phase), std::sin(phase)};
	}

	fers_signal::FileWaveform signal{};
	signal.setKind(fers_signal::FileWaveformKind::Fmcw);
	signal.load(samples, sample_count, rate);
	const RealType duration = static_cast<RealType>(sample_count) / rate;
	const RealType sample_time = 0.3125;
	const auto first = signal.sampleAt(sample_time);
	const auto at_end = signal.sampleAt(duration);
	const auto after_end = signal.sampleAt(sample_time + duration);
	const auto negative = signal.sampleAt(-sample_time);

	REQUIRE_THAT(std::abs(first), WithinAbs(1.0, 1e-6));
	REQUIRE_THAT(std::arg(first), WithinAbs(std::remainder(2.0 * PI * 3.0 * sample_time, 2.0 * PI), 1e-6));
	REQUIRE((at_end == ComplexType{0.0, 0.0}));
	REQUIRE((after_end == ComplexType{0.0, 0.0}));
	REQUIRE((negative == ComplexType{0.0, 0.0}));
}

TEST_CASE("FmcwChirpWaveform applies sweep direction to phase only", "[signal][radar][fmcw]")
{
	const RealType bandwidth = 2.0e6;
	const RealType duration = 1.0e-3;
	const RealType offset = 1.0e5;
	const RealType u = 4.0e-4;

	fers_signal::FmcwChirpWaveform const up(bandwidth, duration, duration, offset);
	fers_signal::FmcwChirpWaveform const down(bandwidth, duration, duration, offset, std::nullopt,
											  fers_signal::FmcwChirpDirection::Down);

	const RealType alpha = bandwidth / duration;
	REQUIRE_FALSE(up.isDownChirp());
	REQUIRE(down.isDownChirp());
	REQUIRE_THAT(up.getChirpRate(), WithinAbs(alpha, 1e-6));
	REQUIRE_THAT(down.getChirpRate(), WithinAbs(alpha, 1e-6));
	REQUIRE_THAT(up.getSignedChirpRate(), WithinAbs(alpha, 1e-6));
	REQUIRE_THAT(down.getSignedChirpRate(), WithinAbs(-alpha, 1e-6));
	REQUIRE_THAT(up.basebandPhaseForChirpTime(u), WithinAbs(2.0 * PI * offset * u + PI * alpha * u * u, 1e-9));
	REQUIRE_THAT(down.basebandPhaseForChirpTime(u), WithinAbs(2.0 * PI * offset * u - PI * alpha * u * u, 1e-9));
}

TEST_CASE("FmcwTriangleWaveform keeps phase continuous at leg and period boundaries", "[signal][radar][fmcw]")
{
	const RealType bandwidth = 2.0e6;
	const RealType duration = 1.0e-3;
	const RealType offset = 1.0e5;
	const RealType alpha = bandwidth / duration;
	const RealType eps = 1.0e-10;

	fers_signal::FmcwTriangleWaveform const triangle(bandwidth, duration, offset, 4);

	REQUIRE_THAT(triangle.getChirpRate(), WithinAbs(alpha, 1e-6));
	REQUIRE_THAT(triangle.getTrianglePeriod(), WithinAbs(2.0 * duration, 1e-15));
	REQUIRE_THAT(triangle.getDeltaPhiUp(),
				 WithinAbs(2.0 * PI * offset * duration + PI * alpha * duration * duration, 1e-6));

	const RealType apex_left = triangle.basebandPhaseForTriangleTime(duration - eps);
	const RealType apex = triangle.basebandPhaseForTriangleTime(duration);
	const RealType apex_right = triangle.basebandPhaseForTriangleTime(duration + eps);
	REQUIRE_THAT(apex, WithinAbs(triangle.getDeltaPhiUp(), 1e-6));
	REQUIRE(std::abs(apex - apex_left) < 2.0);
	REQUIRE(std::abs(apex_right - apex) < 2.0);

	const RealType period_phase = triangle.basebandPhaseForTriangleTime(triangle.getTrianglePeriod());
	REQUIRE_THAT(period_phase, WithinAbs(2.0 * triangle.getDeltaPhiUp(), 1e-6));
	REQUIRE(triangle.instantaneousBasebandPhase(3.5 * triangle.getTrianglePeriod()).has_value());
	REQUIRE_FALSE(triangle.instantaneousBasebandPhase(4.0 * triangle.getTrianglePeriod()).has_value());
}

TEST_CASE("SteppedFrequencyWaveform selects active dwell steps and finite sweeps", "[signal][radar][sfcw]")
{
	fers_signal::SfcwWaveform const sfcw(10.0, 2.0, 4, 0.25, 0.5, std::size_t{2});

	REQUIRE_THAT(sfcw.firstFrequency(100.0), WithinAbs(110.0, 1e-12));
	REQUIRE_THAT(sfcw.lastFrequency(100.0), WithinAbs(116.0, 1e-12));
	REQUIRE_THAT(sfcw.frequencySpan(), WithinAbs(6.0, 1e-12));
	REQUIRE_THAT(sfcw.effectiveBandwidth(), WithinAbs(8.0, 1e-12));
	REQUIRE_THAT(sfcw.getSweepPeriod(), WithinAbs(2.0, 1e-12));
	REQUIRE(sfcw.totalDuration().has_value());
	REQUIRE_THAT(sfcw.totalDuration().value_or(0.0), WithinAbs(4.0, 1e-12));

	const auto first = sfcw.activeStepAt(0.1, 100.0);
	REQUIRE(first.has_value());
	REQUIRE(first.value().sweep_index == 0u);
	REQUIRE(first.value().step_index == 0u);
	REQUIRE_THAT(first.value().rf_frequency, WithinAbs(110.0, 1e-12));
	REQUIRE_THAT(first.value().step_start_time, WithinAbs(0.0, 1e-12));
	REQUIRE_THAT(first.value().dwell_end_time, WithinAbs(0.25, 1e-12));

	REQUIRE_FALSE(sfcw.activeStepAt(0.3, 100.0).has_value());

	const auto second_sweep = sfcw.activeStepAt(2.55, 100.0);
	REQUIRE(second_sweep.has_value());
	REQUIRE(second_sweep.value().sweep_index == 1u);
	REQUIRE(second_sweep.value().step_index == 1u);
	REQUIRE_THAT(second_sweep.value().rf_frequency, WithinAbs(112.0, 1e-12));
	REQUIRE_FALSE(sfcw.activeStepAt(4.0, 100.0).has_value());

	fers_signal::SfcwWaveform const descending(0.0, -5.0, 3, 0.1, 0.2);
	REQUIRE_THAT(descending.lastFrequency(100.0), WithinAbs(90.0, 1e-12));
	REQUIRE_THAT(descending.frequencySpan(), WithinAbs(10.0, 1e-12));
	REQUIRE_FALSE(descending.totalDuration().has_value());
}

TEST_CASE("RadarSignal exposes metadata", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	std::vector<ComplexType> data = {ComplexType{1.0, 0.0}};
	fers_signal::PulseWaveform signal;
	signal.load(data, static_cast<unsigned>(data.size()), 1000.0);
	signal.setFilename("waveform.bin");

	const fers_signal::RadarSignal radar("waveform", 9.0, 77.0, std::move(signal), 42);

	REQUIRE(radar.getName() == "waveform");
	REQUIRE(radar.getCarrier() == 77.0);
	REQUIRE(radar.getPower() == 9.0);
	REQUIRE(radar.getId() == 42);

	const auto* sampled = radar.getPulseWaveform();
	REQUIRE(sampled != nullptr);
	REQUIRE_THAT(sampled->getDuration(), WithinAbs(0.001, 1e-12));
	REQUIRE(sampled->getRate() == 1000.0);
	REQUIRE(sampled->getFilename().has_value());
	REQUIRE(sampled->getFilename().value_or("") == "waveform.bin");
}

TEST_CASE("RadarSignal autogenerates waveform ids", "[signal][radar]")
{
	fers_signal::RadarSignal const radar("waveform", 1.0, 1.0, fers_signal::CwWaveform{}, 0);

	REQUIRE(SimIdGenerator::getType(radar.getId()) == ObjectType::Waveform);
}

TEST_CASE("PulseWaveform render scales output by the amplitude parameter", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);

	// At exactly zero fractional delay the render filter is an exact identity (see
	// test_response.cpp for why), so with delay == 0.0 the rendered output is
	// the loaded samples scaled by the amplitude parameter, no mock needed.
	const std::vector<ComplexType> input = {ComplexType{1.0, 0.0}, ComplexType{2.0, -1.0}};
	fers_signal::PulseWaveform signal;
	signal.load(input, static_cast<unsigned>(input.size()), 1.0);

	const std::vector<interp::InterpPoint> points = {{1.0, 0.0, 0.0, 0.0}};
	const auto data = signal.render(points, points.front().rx_time, 0.0, 2.0);

	REQUIRE(data.size() == input.size());
	REQUIRE_THAT(data[0].real(), WithinAbs(2.0, 1e-9));
	REQUIRE_THAT(data[0].imag(), WithinAbs(0.0, 1e-9));
	REQUIRE_THAT(data[1].real(), WithinAbs(4.0, 1e-9));
	REQUIRE_THAT(data[1].imag(), WithinAbs(-2.0, 1e-9));
}

TEST_CASE("PulseWaveform load applies oversampling to rate and sample count", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(2);
	std::vector<ComplexType> input = {ComplexType{1.0, 0.0}, ComplexType{0.5, -0.5}};

	fers_signal::PulseWaveform signal;
	signal.load(input, static_cast<unsigned>(input.size()), 100.0);

	REQUIRE(signal.getRate() == 200.0);

	const std::vector<interp::InterpPoint> points = {{1.0, 0.0, 0.0, 0.0}};
	const auto data = signal.render(points, points.front().rx_time, 0.0, 1.0);
	REQUIRE(data.size() == input.size() * 2);
	REQUIRE(signal.getSampleCount() == input.size() * 2);
}

TEST_CASE("PulseWaveform render matches constant input physics", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);

	const unsigned filter_length = params::renderFilterLength();
	const unsigned sample_count = filter_length * 4;
	std::vector<ComplexType> input(sample_count, ComplexType{1.0, 0.0});

	fers_signal::PulseWaveform signal;
	signal.load(input, sample_count, 1.0);

	const RealType power = 4.0;
	const RealType phase = PI / 4.0;
	const std::vector<interp::InterpPoint> points = {{std::sqrt(power), 0.0, 0.0, phase}};

	const auto data = signal.render(points, points.front().rx_time, 0.0, 1.0);

	const auto& interp = interp::InterpFilter::getInstance();
	const auto filter = interp.getFilter(0.0);
	const auto expected = expectedConstantRender(filter, std::sqrt(power), phase);

	const unsigned sample_index = filter_length;
	REQUIRE(sample_index < data.size());
	REQUIRE_THAT(data[sample_index].real(), WithinAbs(expected.real(), 1e-6));
	REQUIRE_THAT(data[sample_index].imag(), WithinAbs(expected.imag(), 1e-6));
}

TEST_CASE("PulseWaveform render interpolates power and phase", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);

	const unsigned filter_length = params::renderFilterLength();
	const unsigned sample_count = filter_length * 4;
	std::vector<ComplexType> input(sample_count, ComplexType{1.0, 0.0});

	fers_signal::PulseWaveform signal;
	signal.load(input, sample_count, 1.0);

	const RealType power_a = 1.0;
	const RealType power_b = 9.0;
	const RealType phase_a = 0.0;
	const RealType phase_b = PI / 2.0;
	const std::vector<interp::InterpPoint> points = {{std::sqrt(power_a), 0.0, 0.0, phase_a},
													 {std::sqrt(power_b), 2.0 * filter_length, 0.0, phase_b}};

	const auto data = signal.render(points, points.front().rx_time, 0.0, 1.0);

	const auto& interp = interp::InterpFilter::getInstance();
	const auto filter = interp.getFilter(0.0);

	const RealType bw = 0.5;
	const RealType amplitude = std::lerp(std::sqrt(power_a), std::sqrt(power_b), bw);
	const RealType phase = std::lerp(phase_a, phase_b, bw);
	const auto expected = expectedConstantRender(filter, amplitude, phase);

	const unsigned sample_index = filter_length;
	REQUIRE(sample_index < data.size());
	REQUIRE_THAT(data[sample_index].real(), WithinAbs(expected.real(), 1e-6));
	REQUIRE_THAT(data[sample_index].imag(), WithinAbs(expected.imag(), 1e-6));
}

TEST_CASE("PulseWaveform render responds to fractional delay", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);

	const unsigned filter_length = params::renderFilterLength();
	const unsigned sample_count = filter_length * 4;
	std::vector<ComplexType> input(sample_count, ComplexType{1.0, 0.0});

	fers_signal::PulseWaveform signal;
	signal.load(input, sample_count, 1.0);

	const std::vector<interp::InterpPoint> points = {{1.0, 0.0, 0.0, 0.0}};

	const auto data = signal.render(points, points.front().rx_time, 0.25, 1.0);

	const auto& interp = interp::InterpFilter::getInstance();
	const auto filter = interp.getFilter(0.75);
	const auto expected = expectedConstantRender(filter, 1.0, 0.0);

	const unsigned sample_index = filter_length;
	REQUIRE(sample_index < data.size());
	REQUIRE_THAT(data[sample_index].real(), WithinAbs(expected.real(), 1e-6));
	REQUIRE_THAT(data[sample_index].imag(), WithinAbs(expected.imag(), 1e-6));
}

TEST_CASE("PulseWaveform render does not clip taps near the buffer edge when the delay unwrap is negative",
		  "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);

	const long filter_length = static_cast<long>(params::renderFilterLength());
	const unsigned sample_count = static_cast<unsigned>(filter_length) * 4;
	const std::vector<ComplexType> input(sample_count, ComplexType{1.0, 0.0});

	fers_signal::PulseWaveform signal;
	signal.load(input, sample_count, 1.0);

	// A single control point keeps the delay (and so the fractional/integer split below)
	// identical for every rendered sample, isolating the convolution boundary clamp from any
	// taper or interpolation effects. 0.3 is deliberately a fractional delay whose
	// idelay = round(0.3) = 0 leaves a *positive* remainder (0.3), so fdelay = -(0.3 - 0)
	// floors to int_part = -1: the render loop ends up centering its filter one native sample
	// *before* the nominal index it's rendering (iSampleUnwrap == -1).
	constexpr RealType delay = 0.3;
	const std::vector<interp::InterpPoint> points = {{1.0, 0.0, delay, 0.0}};

	const auto data = signal.render(points, points.front().rx_time, 0.0, 1.0);
	REQUIRE(data.size() == sample_count);

	const long idelay = std::lround(delay);
	RealType fdelay = -(delay - static_cast<RealType>(idelay));
	const RealType int_part = std::floor(fdelay);
	fdelay -= int_part;
	const long i_sample_unwrap = static_cast<long>(int_part);
	REQUIRE(i_sample_unwrap == -1);

	const auto& interp = interp::InterpFilter::getInstance();
	const auto filter = interp.getFilter(fdelay);

	// Ground-truth oracle: sum every filter tap whose data index (the delay-unwrapped center
	// plus the tap offset) actually falls inside the buffer - with no assumption about how the
	// implementation bounds its own loop. This is exactly what `performConvolution` must
	// compute; before the fix its outer loop bounds were derived from the un-unwrapped sample
	// index, which silently dropped in-range taps near the trailing edge of the buffer.
	const auto expectedAt = [&](const long sample_index) {
		const long center = sample_index + i_sample_unwrap;
		ComplexType sum{0.0, 0.0};
		for (long j = -filter_length / 2; j < filter_length / 2; ++j)
		{
			const long idx = center + j;
			if (idx >= 0 && idx < static_cast<long>(sample_count))
			{
				sum += filter[static_cast<std::size_t>(j + filter_length / 2)] * input[static_cast<std::size_t>(idx)];
			}
		}
		return sum;
	};

	// The last few native samples sit closest to the trailing edge and are exactly where a
	// mis-bounded clamp drops valid taps.
	for (long i = static_cast<long>(sample_count) - 3; i < static_cast<long>(sample_count); ++i)
	{
		const auto expected = expectedAt(i);
		REQUIRE_THAT(data[static_cast<std::size_t>(i)].real(), WithinAbs(expected.real(), 1e-9));
		REQUIRE_THAT(data[static_cast<std::size_t>(i)].imag(), WithinAbs(expected.imag(), 1e-9));
	}
}

TEST_CASE("PulseWaveform renderSlice matches cropped full response with padding", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);

	const unsigned sample_count = params::renderFilterLength() * 6;
	std::vector<ComplexType> input(sample_count);
	for (unsigned i = 0; i < sample_count; ++i)
	{
		const RealType phase = 2.0 * PI * static_cast<RealType>(i) / 17.0;
		input[i] = {std::cos(phase), std::sin(phase)};
	}

	fers_signal::PulseWaveform signal;
	signal.load(input, sample_count, 100.0);
	const std::vector<interp::InterpPoint> points = {{1.0, 10.0, 0.0, 0.0}, {0.25, 12.0, 0.0, PI / 3.0}};

	const auto full = signal.render(points, points.front().rx_time, 0.0, 1.0);
	constexpr unsigned crop_start = 31;
	constexpr unsigned crop_count = 47;
	constexpr unsigned pad = 20;
	const RealType slice_start = points.front().rx_time + static_cast<RealType>(crop_start - pad) / 100.0;
	const auto slice =
		signal.renderSlice(points, points.front().rx_time, slice_start, 100.0, crop_count + 2 * pad, 0.0, 1.0);

	REQUIRE(full.size() == sample_count);
	REQUIRE(slice.size() == crop_count + 2 * pad);
	for (unsigned i = 0; i < crop_count; ++i)
	{
		REQUIRE_THAT(slice[pad + i].real(), WithinAbs(full[crop_start + i].real(), 1e-9));
		REQUIRE_THAT(slice[pad + i].imag(), WithinAbs(full[crop_start + i].imag(), 1e-9));
	}
}

TEST_CASE("PulseWaveform renderSlice interpolates onto non-native output grids", "[signal][radar]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);

	constexpr RealType native_rate_hz = 100.0;
	constexpr RealType output_rate_hz = 137.0;
	constexpr RealType tone_hz = 7.0;
	constexpr RealType start_time = 3.0;
	constexpr RealType slice_start_time = start_time + 6.0;
	constexpr std::size_t slice_count = 256;
	constexpr unsigned sample_count = 2048;

	std::vector<ComplexType> input(sample_count);
	for (unsigned i = 0; i < sample_count; ++i)
	{
		const RealType phase = 2.0 * PI * tone_hz * static_cast<RealType>(i) / native_rate_hz;
		input[i] = {std::cos(phase), std::sin(phase)};
	}

	fers_signal::PulseWaveform signal;
	signal.load(input, sample_count, native_rate_hz);
	const std::vector<interp::InterpPoint> points = {{1.0, start_time, 0.0, 0.0}};

	const auto slice =
		signal.renderSlice(points, points.front().rx_time, slice_start_time, output_rate_hz, slice_count, 0.0, 1.0);

	ComplexType correlation{0.0, 0.0};
	RealType output_energy = 0.0;
	RealType reference_energy = 0.0;
	for (std::size_t i = 0; i < slice.size(); ++i)
	{
		const RealType elapsed = slice_start_time - start_time + static_cast<RealType>(i) / output_rate_hz;
		const RealType phase = 2.0 * PI * tone_hz * elapsed;
		const ComplexType expected{std::cos(phase), std::sin(phase)};
		correlation += slice[i] * std::conj(expected);
		output_energy += std::norm(slice[i]);
		reference_energy += std::norm(expected);
	}

	const RealType normalized_correlation = std::abs(correlation) / std::sqrt(output_energy * reference_energy);
	REQUIRE(normalized_correlation > 0.999);
	REQUIRE(std::abs(std::arg(correlation)) < 0.01);
}
