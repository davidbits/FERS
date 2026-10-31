// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <embree4/rtcore.h>
#include <stdexcept>

#include "core/logging.h"

namespace propagation::raytracing::embree
{
	inline void deviceErrorCallback(void* /*user_ptr*/, const RTCError code, const char* message)
	{
		LOG(logging::Level::ERROR, "Embree device error [{}]: {}", static_cast<int>(code),
			message != nullptr ? message : "");
	}

	/**
	 * @brief RAII handle for an Embree device.
	 */
	class Device
	{
	public:
		explicit Device(const char* config) : _device(rtcNewDevice(config))
		{
			if (_device == nullptr)
				throw std::runtime_error("Failed to create Embree device.");
			rtcSetDeviceErrorFunction(_device, deviceErrorCallback, nullptr);
		}
		~Device()
		{
			if (_device != nullptr)
				rtcReleaseDevice(_device);
		}
		Device(const Device&) = delete;
		Device& operator=(const Device&) = delete;
		Device(Device&&) = delete;
		Device& operator=(Device&&) = delete;

		[[nodiscard]] RTCDevice get() const { return _device; }

	private:
		RTCDevice _device;
	};

	/**
	 * @brief RAII handle for an Embree scene.
	 */
	class Scene
	{
	public:
		explicit Scene(RTCDevice device) : _scene(rtcNewScene(device)) {}
		~Scene()
		{
			if (_scene != nullptr)
				rtcReleaseScene(_scene);
		}
		Scene(const Scene&) = delete;
		Scene& operator=(const Scene&) = delete;

		// Moveable so it can be stored in a `std::vector`.
		Scene(Scene&& other) noexcept : _scene(other._scene) { other._scene = nullptr; }
		Scene& operator=(Scene&& other) noexcept
		{
			if (this != &other)
			{
				if (_scene != nullptr)
					rtcReleaseScene(_scene);
				_scene = other._scene;
				other._scene = nullptr;
			}
			return *this;
		}

		[[nodiscard]] RTCScene get() const { return _scene; }

	private:
		RTCScene _scene;
	};

	/**
	 * @brief RAII handle for an Embree geometry.
	 */
	class Geometry
	{
	public:
		Geometry(RTCDevice device, const RTCGeometryType type) : _geometry(rtcNewGeometry(device, type)) {}
		~Geometry()
		{
			if (_geometry != nullptr)
				rtcReleaseGeometry(_geometry);
		}
		Geometry(const Geometry&) = delete;
		Geometry& operator=(const Geometry&) = delete;

		// Moveable so it can be stored in a `std::vector`.
		Geometry(Geometry&& other) noexcept : _geometry(other._geometry) { other._geometry = nullptr; }
		Geometry& operator=(Geometry&& other) noexcept
		{
			if (this != &other)
			{
				if (_geometry != nullptr)
					rtcReleaseGeometry(_geometry);
				_geometry = other._geometry;
				other._geometry = nullptr;
			}
			return *this;
		}

		[[nodiscard]] RTCGeometry get() const { return _geometry; }

	private:
		RTCGeometry _geometry;
	};
}
