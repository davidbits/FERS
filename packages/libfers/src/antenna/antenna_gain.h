// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file antenna_gain.h
 *
 * @brief Antenna gain-pattern formulas, shared between the CPU antenna model
 * (`antenna::Antenna` subclasses) and the GPU ray-tracing propagation model.
 *
 * These functions are templated by precision as GPUs benefit greatly from 32-bit precision,
 * and double-precision accuracy is not critical for antenna gains.
 * Note that the bessel J1 function is not available on GPUs. A Taylor approximated version is used.
 *
 * Functions are plain `constexpr` free function operating only on scalar
 * angles and antenna parameters to be portable to CUDA/HIP device code.
 * This works because plain, unannotated `constexpr` functions are callable from both host and
 * device code by `nvcc` with `--expt-relaxed-constexpr` (see `cmake/FersOptiX.cmake`).
 */

#pragma once

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define DEVICE 1
#else
#define DEVICE 0
#endif

#include <cmath>
#include <limits>

#if !DEVICE
#include "core/portable_utils.h"
#endif

namespace antenna::gain
{
	/// The ratio of a circle's circumference to its diameter.
	// Far more than enough digits to saturate double-precision.
	template <typename Real>
	constexpr Real PI = Real(3.1415926535897932384626433832795028841971693993751);

	/// sinc(x) = sin(x)/x, with the x=0 singularity resolved to its limit, 1.
	template <typename Real>
	constexpr Real sinc(const Real x) noexcept
	{
		if (std::abs(x) < std::numeric_limits<Real>::epsilon())
		{
			return Real(1);
		}
		return std::sin(x) / x;
	}

	/// `antenna::Sinc`'s gain pattern (excluding the efficiency factor, applied by the caller).
	template <typename Real>
	constexpr Real sincGain(const Real theta, const Real alpha, const Real beta, const Real gamma) noexcept
	{
		return alpha * std::pow(std::abs(sinc(beta * theta)), gamma);
	}

	/// `antenna::Gaussian`'s gain pattern (excluding the efficiency factor, applied by the caller).
	template <typename Real>
	constexpr Real gaussianGain(const Real delta_azimuth, const Real delta_elevation, const Real azimuth_scale,
								const Real elevation_scale) noexcept
	{
		return std::exp(-delta_azimuth * delta_azimuth * azimuth_scale) *
			std::exp(-delta_elevation * delta_elevation * elevation_scale);
	}

	/// `antenna::SquareHorn`'s gain pattern (excluding the efficiency factor, applied by the caller).
	template <typename Real>
	constexpr Real squareHornGain(const Real theta, const Real dimension, const Real wavelength) noexcept
	{
		const Real directivity = Real(4) * PI<Real> * dimension * dimension / (wavelength * wavelength);
		const Real x = PI<Real> * dimension * std::sin(theta) / wavelength;
		const Real s = sinc(x);
		return directivity * s * s;
	}

	/**
	 * @brief A portable rational approximation of the Bessel function of the first kind, order 1,
	 * J1(x), accurate to roughly 1e-8 relative error. Used in place of `core::besselJ1` for deivce code.
	 *
	 * Split-domain rational-polynomial form with Abramowitz & Stegun 9.4 coefficients.
	 */
	template <typename Real>
	constexpr Real besselJ1Approx(const Real x) noexcept
	{
		const Real abs_x = std::abs(x);
		if (abs_x < Real(8))
		{
			const Real y = x * x;
			const Real p = x *
				(Real(72362614232.0) +
				 y *
					 (Real(-7895059235.0) +
					  y *
						  (Real(242396853.1) +
						   y * (Real(-2972611.439) + y * (Real(15704.48260) + y * Real(-30.16036606))))));
			const Real q = Real(144725228442.0) +
				y *
					(Real(2300535178.0) +
					 y *  (Real(18583304.74) +  y * (Real(99447.43394) + y * (Real(376.9991397) + y * Real(1.0)))));
			return p / q;
		}

		const Real z = Real(8) / abs_x;
		const Real y = z * z;
		const Real envelope_phase = abs_x - Real(2.356194491);
		const Real p = Real(1.0) +
			y *
				(Real(0.183105e-2) +
				 y * (Real(-0.3516396496e-4) + y * (Real(0.2457520174e-5) + y * Real(-0.240337019e-6))));
		const Real q = Real(0.04687499995) +
			y *
				(Real(-0.2002690873e-3) +
				 y * (Real(0.8449199096e-5) + y * (Real(-0.88228987e-6) + y * Real(0.105787412e-6))));
		Real result =
			std::sqrt(Real(0.636619772) / abs_x) * (std::cos(envelope_phase) * p - z * std::sin(envelope_phase) * q);
		if (x < Real(0))
		{
			result = -result;
		}
		return result;
	}

	/// J1(x)/x, with the x=0 singularity resolved to its limit, 1/2.
	template <typename Real>
	constexpr Real besselJ1OverX(const Real x) noexcept
	{
		if (std::abs(x) < std::numeric_limits<Real>::epsilon())
		{
			return Real(0.5);
		}

#if DEVICE
		const Real bessel_j1 = besselJ1Approx(x);
#else
		const Real bessel_j1 = Real(core::besselJ1(double(x)));
#endif

		return bessel_j1 / x;
	}

	/// `antenna::Parabolic`'s gain pattern (excluding the efficiency factor, applied by the caller).
	template <typename Real>
	constexpr Real parabolicGain(const Real theta, const Real diameter, const Real wavelength) noexcept
	{
		const Real k = PI<Real> * diameter / wavelength;
		const Real directivity = k * k;
		const Real x = k * std::sin(theta);
		const Real j1_over_x = besselJ1OverX(x);
		return directivity * (Real(2) * j1_over_x) * (Real(2) * j1_over_x);
	}
}
