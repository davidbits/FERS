// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file assets.h
 * @brief Lightweight, non-polymorphic asset descriptors owned by the World.
 *
 * These are scenario-level asset *references*, not loaded geometry data: `MeshAsset` holds a
 * resolved filesystem path, not vertex/triangle data, and `MaterialAsset` holds the material
 * parameters directly. Propagation models that need the actual mesh data (e.g.
 * `propagation::raytracing::Mesh`) load it themselves, on demand, from the path here.
 */

#pragma once

#include <filesystem>
#include <string>

#include "core/sim_id.h"

namespace core
{
	/**
	 * @struct MeshAsset
	 * @brief A named reference to a mesh file on disk.
	 */
	struct MeshAsset
	{
		SimId id; ///< Unique ID for this mesh asset.
		std::string name; ///< The name used by `<geometry mesh="...">` references.
		std::filesystem::path path; ///< Fully resolved filesystem path to the mesh file.

		/// The original `<mesh filename="...">` attribute, relative to the scenario's "meshes"
		/// directory. Kept alongside the resolved `path` so the XML writer can round-trip it
		/// without needing to know the scenario's base directory.
		std::string filename;
	};

	/**
	 * @struct MaterialAsset
	 * @brief A named surface material definition, used by the ray-tracing PO/GO model's
	 * reflection/current calculations.
	 */
	struct MaterialAsset
	{
		SimId id; ///< Unique ID for this material asset.
		std::string name; ///< The name used by `<geometry material="...">` references.

		/// The relative permittivity of the material.
		///
		/// This is used in conjuction with the `conductivity` to compute the
		/// complex index of refraction: n = sqrt(ε_r' - j·σ/(ω·ε0))
		float relative_permittivity;

		/// The conductivity of the material, Siemens per meter.
		///
		/// Zero for dielectrics, infinity for perfect electric conductors.
		float conductivity;
	};
}
