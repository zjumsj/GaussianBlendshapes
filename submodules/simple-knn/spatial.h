/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>

/* GPU operator */
torch::Tensor distCUDA2(const torch::Tensor& points);

std::vector<torch::Tensor> getIdAndWeight(
    const torch::Tensor& in_points, // Nx3
    const torch::Tensor& grid, // DHW
    bool align_corners
);

/* CPU operator */
torch::Tensor getNearestFeature(
    const torch::Tensor& in_points, // Nx3
    const torch::Tensor& point_features, // NxC
    const torch::Tensor& faces, // Fx3, int32
    const torch::Tensor& query_points, // Mx3
    int32_t findNumber
);
torch::Tensor getNearestTriangleID(
    const torch::Tensor& in_points, // Nx3
    const torch::Tensor& faces, // Fx3
    const torch::Tensor& query_points, // Mx3
    int32_t findNumber
);
