// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cmath>
#include <cstdint>

#ifndef HC_FN
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HC_FN __host__ __device__ inline
#else
#define HC_FN inline
#endif
#endif

namespace propagation
{
	/// The ratio of a circle's circumference to its diameter.
	// Far more than enough digits to saturate double-precision.
	template <typename Real = double>
	constexpr Real PI_V = Real(3.1415926535897932384626433832795028841971693993751);

	/// Intrinsic impedance of free space, Ohms.
	/// https://physics.nist.gov/cgi-bin/cuu/Value?z0
	/// Accessed 2026-09-12
	constexpr float Z0 = 376.730313412f;

	/// Vacuum electric permittivity, in Farads/meter.
	/// https://physics.nist.gov/cgi-bin/cuu/Value?ep0
	/// Accessed 2026-09-15
	constexpr float E0 = 8.8541878188e-12f;

	/// Speed of light in a vacuum. Meters per second.
	/// https://physics.nist.gov/cgi-bin/cuu/Value?c
	/// Accessed 2026-09-15
	template <typename Real>
	constexpr Real C = Real(299792458);

	// Don't use HLSL/CUDA/HIP naming convention to mitigate naming clashes and confusion.
	// They use double3/float2/uint4 etc.
	// We'll use PascalCase versions to distinguish our own types, inline with FERS naming convention.

	template <typename Real>
	struct Vec2
	{
		Real x, y;
		HC_FN constexpr Vec2() : x(), y() {}
		HC_FN constexpr Vec2(Real x_, Real y_) : x(x_), y(y_) {}
	};

	template <typename Real>
	struct Vec3
	{
		Real x, y, z;
		HC_FN constexpr Vec3() : x(), y(), z() {}
		HC_FN constexpr Vec3(Real x_, Real y_, Real z_) : x(x_), y(y_), z(z_) {}

		/// Elementwise converting constructor, e.g. for narrowing a `Double3` to a `Float3`.
		template <typename U>
		HC_FN constexpr explicit Vec3(const Vec3<U>& v) :
			x(static_cast<Real>(v.x)), y(static_cast<Real>(v.y)), z(static_cast<Real>(v.z))
		{
		}
	};

	template <typename Real>
	HC_FN constexpr Vec3<Real> operator+(const Vec3<Real>& a, const Vec3<Real>& b)
	{
		return Vec3<Real>{a.x + b.x, a.y + b.y, a.z + b.z};
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real>& operator+=(Vec3<Real>& a, Vec3<Real> b)
	{
		a = a + b;
		return a;
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real> operator-(const Vec3<Real>& a, const Vec3<Real>& b)
	{
		return Vec3<Real>{a.x - b.x, a.y - b.y, a.z - b.z};
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real> operator-(const Vec3<Real>& a)
	{
		return Vec3<Real>{-a.x, -a.y, -a.z};
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real> operator*(const Vec3<Real>& a, Real s)
	{
		return Vec3<Real>{a.x * s, a.y * s, a.z * s};
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real> operator*(Real s, const Vec3<Real>& a)
	{
		return a * s;
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real>& operator*=(Vec3<Real>& a, Real s)
	{
		a = a * s;
		return a;
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real> operator/(const Vec3<Real>& a, Real s)
	{
		return Vec3<Real>{a.x / s, a.y / s, a.z / s};
	}
	template <typename Real>
	HC_FN constexpr Real dot(const Vec3<Real>& a, const Vec3<Real>& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	template <typename Real>
	HC_FN constexpr Vec3<Real> cross(const Vec3<Real>& a, const Vec3<Real>& b)
	{
		return Vec3<Real>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
	}
	template <typename Real>
	HC_FN constexpr Real length2(const Vec3<Real>& a)
	{
		return dot(a, a);
	}
	template <typename Real>
	HC_FN Real length(const Vec3<Real>& a)
	{
		return std::sqrt(length2(a));
	}
	template <typename Real>
	HC_FN Vec3<Real> normalize(const Vec3<Real>& a)
	{
		return a / length(a);
	}

	using Uint3 = Vec3<uint32_t>;

	template <typename Real>
	using Real2 = Vec2<Real>;

	template <typename Real>
	using Real3 = Vec3<Real>;
	using Float3 = Real3<float>;
	using Double3 = Real3<double>;

	/// A 2x2 matrix, stored as two column vectors.
	template <typename Real>
	struct Mat2x2
	{
		Vec2<Real> x, y;
	};

	template <typename Real>
	using Real2x2 = Mat2x2<Real>;

	/// A 3x4 matrix, stored as four column vectors.
	///
	/// Used for affine (rotation + translation) local<->world transforms. See `amul`/`amuld`.
	template <typename Real>
	struct Mat3x4
	{
		Vec3<Real> x, y, z, w;
	};

	using Float3x4 = Mat3x4<float>;

	template <typename Real>
	struct Complex
	{
		Real re, im;
		HC_FN constexpr Complex() : re(0.), im(0.) {}
		HC_FN constexpr Complex(Real r, Real i = 0.) : re(r), im(i) {}
	};

	template <typename Real>
	HC_FN constexpr Complex<Real> operator+(Complex<Real> a, Complex<Real> b)
	{
		return Complex(a.re + b.re, a.im + b.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real>& operator+=(Complex<Real>& a, Complex<Real> b)
	{
		a = a + b;
		return a;
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator-(Complex<Real> a, Complex<Real> b)
	{
		return Complex(a.re - b.re, a.im - b.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator-(Real a, Complex<Real> b)
	{
		return Complex(a - b.re, -b.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator-(Complex<Real> a)
	{
		return Complex(-a.re, -a.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator*(Complex<Real> a, Complex<Real> b)
	{
		return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator*(Complex<Real> a, Real s)
	{
		return Complex(a.re * s, a.im * s);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator*(Real s, Complex<Real> a)
	{
		return a * s;
	}
	template <typename Real>
	HC_FN constexpr Complex<Real>& operator*=(Complex<Real>& a, Real s)
	{
		a = a * s;
		return a;
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator/(Complex<Real> a, Complex<Real> b)
	{
		Real d = b.re * b.re + b.im * b.im;
		return Complex((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> operator/(Complex<Real> a, Real s)
	{
		return Complex(a.re / s, a.im / s);
	}
	template <typename Real>
	HC_FN constexpr Real cabs(Complex<Real> a)
	{
		return std::sqrt(a.re * a.re + a.im * a.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> conj(Complex<Real> a)
	{
		return Complex(a.re, -a.im);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> mul_i(Complex<Real> a) // a * j
	{
		return Complex(-a.im, a.re);
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> cexp_i(Real x) // e^{jx}
	{
		return Complex<Real>(std::cos(x), std::sin(x));
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> csqrt(Complex<Real> a)
	{
		if (a.re == Real(0.0) && a.im == Real(0.0))
			return Complex<Real>(0.0, 0.0);

		Real r = cabs(a);

		if (a.re >= Real(0.0))
		{
			Real w = std::sqrt(Real(0.5) * (r + a.re)); // safety: sum of positive terms
			return Complex<Real>(w, a.im / (Real(2.0) * w));
		}

		Real w = std::sqrt(Real(0.5) * (r - a.re)); // safety: positive minus negative
		Real im = (a.im >= Real(0.0)) ? w : -w;
		return Complex<Real>(a.im / (Real(2.0) * im), im);
	}

	using CFloat = Complex<float>;
	using CDouble = Complex<double>;


	template <typename Real>
	struct Complex2
	{
		Complex<Real> x, y;
	};

	template <typename Real>
	HC_FN constexpr Complex2<Real> operator+(Complex2<Real> a, Complex2<Real> b)
	{
		return Complex2{a.x + b.x, a.y + b.y};
	}
	template <typename Real>
	HC_FN constexpr Complex2<Real> operator-(const Complex2<Real>& v)
	{
		return Complex2{-v.x, -v.y, -v.z};
	}
	template <typename Real>
	HC_FN constexpr Complex2<Real> operator*(const Real2<Real>& r, Complex<Real> c)
	{
		return Complex2{r.x * c, r.y * c, r.z * c};
	}
	template <typename Real>
	HC_FN constexpr Complex2<Real> operator*(Complex<Real> c, const Real2<Real>& r)
	{
		return r * c;
	}
	template <typename Real>
	HC_FN constexpr Complex2<Real> operator*(const Real2x2<Real>& m, Complex2<Real> v)
	{
		return Complex2{m.x.x * v.x + m.y.x * v.y, m.x.y * v.x + m.y.y * v.y};
	}

	template <typename Real>
	struct Complex3
	{
		Complex<Real> x, y, z;
	};

	template <typename Real>
	HC_FN constexpr Complex3<Real> operator+(Complex3<Real> a, Complex3<Real> b)
	{
		return Complex3{a.x + b.x, a.y + b.y, a.z + b.z};
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator-(const Complex<Real>& v)
	{
		return Complex3{-v.x, -v.y, -v.z};
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator-(Complex3<Real> a, Complex3<Real> b)
	{
		return Complex3{a.x - b.x, a.y - b.y, a.z - b.z};
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator*(const Real3<Real>& r, Complex<Real> c)
	{
		return Complex3{r.x * c, r.y * c, r.z * c};
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator*(Complex<Real> c, const Real3<Real>& r)
	{
		return r * c;
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator*(const Complex3<Real>& v, Complex<Real> c)
	{
		return Complex3{v.x * c, v.y * c, v.z * c};
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator*(Complex<Real> c, const Complex3<Real>& v)
	{
		return v * c;
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator*(const Complex3<Real>& v, Real s)
	{
		return Complex3{v.x * s, v.y * s, v.z * s};
	}
	template <typename Real>
	HC_FN constexpr Complex3<Real> operator/(const Complex3<Real>& v, Real s)
	{
		return Complex3{v.x / s, v.y / s, v.z / s};
	}

	template <typename Real>
	HC_FN constexpr Complex3<Real> cross(const Real3<Real>& a, const Complex3<Real>& b)
	{
		return Complex3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> dot(const Complex3<Real>& a, const Real3<Real>& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	template <typename Real>
	HC_FN constexpr Complex<Real> dot_no_conj(const Complex3<Real>& a, const Complex3<Real>& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	using CFloat3 = Complex3<float>;
	using CDouble3 = Complex3<double>;


	// --- Angles and Rotations --------------------------------------------------------------------- //

	/**
	 * @brief A pure azimuth/elevation orientation (no roll).
	 *
	 * Follows the same convention as `Vec3(SVec3)`: azimuth=0, elevation=0 means
	 * local +X; `rotateLocalToWorld(AzEl{0,0}, ...)`'s boresight direction is
	 * `(cos(az)cos(el), sin(az)cos(el), sin(el))`, i.e. `Vec3(SVec3(1, az, el))`.
	 */
	template <typename Real>
	struct AzEl
	{
		Real azimuth{};
		Real elevation{};
	};

	using FloatAzEl = AzEl<float>;
	using DoubleAzEl = AzEl<double>;

	/**
	 * @brief Rotates a local direction (in the same az/el spherical convention as `SVec3` - see
	 * `AzEl`) into world space, given the frame's own world-space orientation `rot`.
	 *
	 * Equivalent to constructing the rotation matrix R = Rz(rot.azimuth) * Ry(-rot.elevation) (the
	 * "yaw then pitch, no roll" rotation that carries local +X to
	 * `Vec3(SVec3(1, rot.azimuth, rot.elevation))`) and applying it to `local_dir`.
	 */
	template <typename Real>
	HC_FN Real3<Real> rotateLocalToWorld(const AzEl<Real>& rot, const Real3<Real>& local_dir)
	{
		const Real caz = std::cos(rot.azimuth), saz = std::sin(rot.azimuth);
		const Real cel = std::cos(rot.elevation), sel = std::sin(rot.elevation);

		return Real3<Real>{caz * cel * local_dir.x - saz * local_dir.y - caz * sel * local_dir.z,
						   saz * cel * local_dir.x + caz * local_dir.y - saz * sel * local_dir.z,
						   sel * local_dir.x + cel * local_dir.z};
	}

	/**
	 * @brief Inverse of `rotateLocalToWorld`: expresses a world-space direction in the frame's local
	 * coordinates. Since the rotation is orthonormal, this is just its transpose applied to
	 * `world_dir`.
	 */
	template <typename Real>
	HC_FN Real3<Real> rotateWorldToLocal(const AzEl<Real>& rot, const Real3<Real>& world_dir)
	{
		const Real caz = std::cos(rot.azimuth), saz = std::sin(rot.azimuth);
		const Real cel = std::cos(rot.elevation), sel = std::sin(rot.elevation);

		return Real3<Real>{caz * cel * world_dir.x + saz * cel * world_dir.y + sel * world_dir.z,
						   -saz * world_dir.x + caz * world_dir.y,
						   -caz * sel * world_dir.x - saz * sel * world_dir.y + cel * world_dir.z};
	}

	// --- Affine local<->global space geometry transforms ------------------------------------------- //

	/// Applies a 3x4 affine matrix to a 3-vector.
	/// Result = A * v + t
	///
	/// This takes floats, converts to doubles, and outputs doubles.
	/// This is designed for converting from single-precision local space to
	/// double-precision world space.
	HC_FN Double3 amuld(const Float3x4& m, const Float3& v) noexcept
	{
		const double vx = static_cast<double>(v.x), vy = static_cast<double>(v.y), vz = static_cast<double>(v.z);

		double x = static_cast<double>(m.w.x);
		x = fma(static_cast<double>(m.x.x), vx, x);
		x = fma(static_cast<double>(m.y.x), vy, x);
		x = fma(static_cast<double>(m.z.x), vz, x);

		double y = static_cast<double>(m.w.y);
		y = fma(static_cast<double>(m.x.y), vx, y);
		y = fma(static_cast<double>(m.y.y), vy, y);
		y = fma(static_cast<double>(m.z.y), vz, y);

		double z = static_cast<double>(m.w.z);
		z = fma(static_cast<double>(m.x.z), vx, z);
		z = fma(static_cast<double>(m.y.z), vy, z);
		z = fma(static_cast<double>(m.z.z), vz, z);


		return {x, y, z};
	}
}
