// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <string>
#include <vector>

#include "propagation/math.h"
#include "propagation/raytracing/sbr_shared.h"

namespace propagation::raytracing
{
	TriangleMeshView loadMesh(const std::string& file_path, std::vector<Float3>& vertices, std::vector<Uint3>& indices);
}
