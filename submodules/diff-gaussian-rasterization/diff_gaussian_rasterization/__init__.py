#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

class _CompositeBlendShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, blendshape, coefficient, init_blendshape, mask):
        # grad towards these parameters is currently not supported
        assert(coefficient.requires_grad == False)
        if init_blendshape is not None:
            assert(init_blendshape.requires_grad == False)
        if mask is not None:
            assert(mask.requires_grad == False)

        mean_shape = list(mean.shape)
        blendshape_shape = list(blendshape.shape)
        ctx.dims = [mean_shape, blendshape_shape]
        ctx.save_for_backward(coefficient,mask)
        output =  _C.CompositeBlendShape(mean, blendshape, coefficient, init_blendshape, mask)
        output = output.view(mean_shape)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        mean_shape, blendshape_shape = ctx.dims
        coefficient, mask = ctx.saved_tensors
        grad_mean, grad_blendshape = _C.CompositeBlendShape_backward(
            grad_out, coefficient, mask
        )
        grad_mean = grad_mean.view(mean_shape)
        grad_blendshape = grad_blendshape.view(blendshape_shape)
        return grad_mean, grad_blendshape, None, None, None

def CompositeBlendShape(mean,blendshape,coefficient,init_blendshape=None,mask=None):
    return _CompositeBlendShape.apply(mean,blendshape,coefficient,init_blendshape,mask)


class _InvMat3x3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat):
        mat = mat.contiguous()
        ctx.save_for_backward(mat)
        return _C.InvMat3x3(mat)
    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        mat,  = ctx.saved_tensors
        grad_in = _C.InvMat3x3_backward(
            mat, grad
        )
        return grad_in

InvMat3x3 = _InvMat3x3.apply

class _RotSH3_C3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rot, sh):
        rot = rot.contiguous()
        sh = sh.contiguous()
        flag = rot.requires_grad
        assert(rot.shape[-1]==3 and rot.shape[-2] == 3)
        assert(sh.shape[-1]==3 and sh.shape[-2]==16 )
        if flag:
            ctx.save_for_backward(rot,sh)
        else:
            ctx.save_for_backward(rot)
        ctx.dims = [flag]
        return _C.RotSH3_C3(
            rot,sh
        )

    @staticmethod
    def backward(ctx, grad):
        grad_rot = None
        grad_sh = None
        grad = grad.contiguous()
        flag, = ctx.dims
        if flag:
            rot,sh = ctx.saved_tensors
            grad_rot, grad_sh = _C.RotSH3_C3_backward(
                rot, sh, grad
            )
        else:
            rot, = ctx.saved_tensors
            grad_sh = _C.RotSH3_C3_simple_backward(
                rot, grad
            )
        return grad_rot, grad_sh

RotSH3_C3 = _RotSH3_C3.apply

class _RotSHBlendShape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expr, rot_q, grid_id, grid_w, sh):
        expr = expr.contiguous()
        rot_q = rot_q.contiguous()
        grid_id = grid_id.contiguous()
        grid_w = grid_w.contiguous()
        sh = sh.contiguous()
        ctx.save_for_backward(expr,rot_q,grid_id,grid_w)
        return _C.RotSHBlendShape(
            expr, rot_q, grid_id, grid_w, sh
        )

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        expr, rot_q, grid_id, grid_w = ctx.saved_tensors
        grad_sh = _C.RotSHBlendShape_backward(
            expr, rot_q, grid_id, grid_w, grad
        )
        return None, None, None, None, grad_sh

RotSHBlendShape = _RotSHBlendShape.apply

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    alpha_point_id
):
    if alpha_point_id is not None:
        return _RasterizeGaussiansAlpha.apply(
            alpha_point_id,
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
        )
    else:
        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
        )

class _RasterizeGaussiansAlpha(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        alpha_point_id,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            alpha_point_id,
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, alpha0, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_alpha(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, alpha0, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_alpha(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        ctx.dims = [alpha_point_id]
        return color, alpha0, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_alpha0, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors
        alpha_point_id = ctx.dims[0]

        # Restructure args as C++ method expects them
        args = (
                alpha_point_id,
                raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_alpha0,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_alpha_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_alpha_backward(*args)

        grads = (
            None,
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, alpha_point_id = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            alpha_point_id
        )

