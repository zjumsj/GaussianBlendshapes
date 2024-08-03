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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

// rot {P}x3x3
// sh {P}xLx3
torch::Tensor RotSH3_C3(
    const torch::Tensor & rot,
    const torch::Tensor & sh
);

// grad_of_sh {P}xLx3
std::tuple<torch::Tensor, torch::Tensor> RotSH3_C3_backward(
    const torch::Tensor & rot,
    const torch::Tensor & sh,
    const torch::Tensor & grad
);

// rot {P}x3x3
// grad {P}xLx3
torch::Tensor RotSH3_C3_simple_backward(
    const torch::Tensor & rot,
    const torch::Tensor & grad
);

torch::Tensor RotSHBlendShape(
    const torch::Tensor & expr, // N
    const torch::Tensor & rot_q, // Fx4xN
    const torch::Tensor & grid_id, // Px8
    const torch::Tensor & grid_w, // Px8
    const torch::Tensor & sh // Px16x3
);

torch::Tensor RotSHBlendShape_backward(
    const torch::Tensor & expr, // N
    const torch::Tensor & rot_q, // Fx4xN
    const torch::Tensor & grid_id, // Px8
    const torch::Tensor & grid_w, // Px8
    const torch::Tensor & grad // Px16x3
);

torch::Tensor InvMat3x3(
    const torch::Tensor & mat // Bx3x3
);
torch::Tensor InvMat3x3_backward(
    const torch::Tensor & mat, // Bx3x3
    const torch::Tensor & grad_out // Bx3x3
);

torch::Tensor CompositeBlendShape(
    const torch::Tensor & mean, // Px?
    const torch::Tensor & blendshape, // Px?xK
    const torch::Tensor & coefficient, // K
    at::optional<at::Tensor> & init_blendshape, // Px?xK
    at::optional<at::Tensor> & mask // PxK
);

std::vector<torch::Tensor> CompositeBlendShape_backward(
    const torch::Tensor & grad_out, // Px?
    //const torch::Tensor & blendshape, // Px?xK
    const torch::Tensor & coefficient, // K
    at::optional<at::Tensor> & mask // PxK
);

void AdamUpdate(
    float beta1, float beta2, float bias_correction1, float sqrt_bias_correction2,
    float step_size, float eps, float weight_decay,
    torch::Tensor param, torch::Tensor grad, torch::Tensor exp_avg, torch::Tensor exp_avg_sq
);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA_Alpha(
    const int point_id0, // start id to accumulate opacity
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA_Alpha(
    const int point_id0, // start id to accumulate opacity
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dT0s,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);