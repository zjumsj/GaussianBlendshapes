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
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);

  m.def("rasterize_gaussians_alpha", &RasterizeGaussiansCUDA_Alpha);
  m.def("rasterize_gaussians_alpha_backward", &RasterizeGaussiansBackwardCUDA_Alpha);

  m.def("RotSH3_C3", &RotSH3_C3);
  m.def("RotSH3_C3_simple_backward", &RotSH3_C3_simple_backward);
  m.def("RotSH3_C3_backward", &RotSH3_C3_backward);

  m.def("RotSHBlendShape", &RotSHBlendShape);
  m.def("RotSHBlendShape_backward", &RotSHBlendShape_backward);

  m.def("InvMat3x3", &InvMat3x3);
  m.def("InvMat3x3_backward", &InvMat3x3_backward);

  m.def("CompositeBlendShape", &CompositeBlendShape);
  m.def("CompositeBlendShape_backward", &CompositeBlendShape_backward);

  m.def("AdamUpdate", &AdamUpdate);
}