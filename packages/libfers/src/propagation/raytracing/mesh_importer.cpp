// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "mesh_importer.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cassert>
#include <cstdint>
#include <string>

#include "propagation/math.h"

namespace propagation::raytracing
{
	TriangleMeshView loadMesh(const std::string& file_path, std::vector<Float3>& vertices, std::vector<Uint3>& indices)
	{
		Assimp::Importer importer;

		// Load the mesh. Post-processing options:
		// aiProcess_Triangulate - Turn polygons in the mesh into traingles. Need this for hardware RT.
		// NO aiProcess_GenNormals/aiProcess_GenSmoothNormals - We generate our own per-facet normals.
		// aiProcess_ValidateDataStructure - Catch obviously broken imports.
		// aiProcess_FindDegenerates - Remove zero-area triangles.
		// aiProcess_FindInvalidData - Avoid NaN/Inf mesh data.
		// aiProcess_SortByPType - Sort points/lines/triangles.
		auto flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_ValidateDataStructure |
			aiProcess_FindDegenerates | aiProcess_FindInvalidData | aiProcess_SortByPType;
		// Potentially useful in the future for materials:
		// aiProcess_GenUVCoords
		// aiProcess_TransformUVCoords

		const auto* scene = importer.ReadFile(file_path, flags);

		if (scene == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0 || scene->mRootNode == nullptr)
		{
			throw std::runtime_error(importer.GetErrorString());
		}

		// TODO_SHAUN maybe address this a bit better.
		// What should we do if a target defines multiple neshes?
		auto* ai_mesh = scene->mMeshes[0];

		TriangleMeshView mesh{};
		mesh.vertex_count = ai_mesh->mNumVertices;
		mesh.triangle_count = ai_mesh->mNumFaces;

		// Compute the index buffer offsets before appending to the vectors.
		mesh.index_buffer_index = static_cast<uint32_t>(indices.size());
		const auto vertex_buffer_offset = static_cast<uint32_t>(vertices.size());

		vertices.reserve(vertices.size() + mesh.vertex_count);
		indices.reserve(indices.size() + mesh.triangle_count);

		for (unsigned int i = 0; i < ai_mesh->mNumVertices; i++)
		{
			auto ai_vertex = ai_mesh->mVertices[i];
			vertices.emplace_back(ai_vertex.x, ai_vertex.y, ai_vertex.z);
		}

		for (unsigned int i = 0; i < ai_mesh->mNumFaces; i++)
		{
			const aiFace face = ai_mesh->mFaces[i];
			assert(face.mNumIndices == 3); // After aiProcess_Triangulate, always 3 indices
			indices.emplace_back(vertex_buffer_offset + face.mIndices[0], vertex_buffer_offset + face.mIndices[1],
								 vertex_buffer_offset + face.mIndices[2]);
		}

		return mesh;
	}
}
