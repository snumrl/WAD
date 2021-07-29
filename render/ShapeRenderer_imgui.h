//
// Created by lasagnaphil on 20. 9. 21..
//

#ifndef MSS_SHAPERENDERER_H
#define MSS_SHAPERENDERER_H

#include <dart/dart.hpp>
#include <unordered_map>
#include "Muscle.h"

using namespace dart::dynamics;
namespace MASS
{

struct ShapeRenderer {
    std::unordered_map<const MeshShape*, std::vector<uint32_t>> meshShapeVbo;
    std::unordered_map<const Muscle*, std::pair<uint32_t, uint32_t>> muscleVboIbo;

    void renderMuscle(Muscle* muscle);

    void renderMesh(const MeshShape* meshShape, bool drawShadows = false, float shadowY = 0.0f,
                    const Eigen::Vector4d& color = Eigen::Vector4d(0.8, 0.8, 0.8, 1.0));
    
    void clearMesh(const MeshShape* meshShape);

private:
    void createMeshVboRecursive(const aiScene* scene, std::vector<uint32_t>& vbo,
                                const aiNode* node);
    void renderMeshRecursive(const aiScene* scene, const std::vector<uint32_t>& vbo,
                             const aiNode* node, int& vboIdx);
};

}
#endif //MSS_SHAPERENDERER_H