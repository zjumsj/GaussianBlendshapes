import torch
import math

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .face_gs_model import GaussianModel
from utils.sh_utils import eval_sh

from typing import List

def render(viewpoint_camera, pcs: List[GaussianModel], pipe,
           bg_color: torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    ## merge sub point cloud to form full point clouds
    n = len(pcs)
    xyz = []
    features = []
    scaling = []
    rotation = []
    opacity = []
    cov3D_precomp_ = []
    for ii in range(n):
        xyz.append(pcs[ii].get_xyz)
        features.append(pcs[ii].get_features)
        opacity.append(pcs[ii].get_opacity)
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        if pipe.compute_cov3D_python:
            cov3D_precomp_.append(pcs[ii].get_covariance(scaling_modifier))
        else:
            scaling.append(pcs[ii].get_scaling)
            rotation.append(pcs[ii].get_rotation)

    if override_color is not None:
        if isinstance(override_color,list) or isinstance(override_color,tuple):
            override_color = torch.cat(override_color,dim=0)

    scales = None
    rotations = None
    cov3D_precomp = None

    xyz = torch.cat(xyz,dim=0)
    features = torch.cat(features,dim=0)
    if pipe.compute_cov3D_python:
        cov3D_precomp = torch.cat(cov3D_precomp_, dim=0)
    else:
        scales = torch.cat(scaling,dim=0)
        rotations = torch.cat(rotation,dim=0)

    opacity = torch.cat(opacity,dim=0)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pcs[0].active_sh_degree, # TODO, all pc should agree
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    #opacity = opacity

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1,2).view(-1,3, (pcs[0].max_sh_degree + 1) ** 2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1,keepdim=True)
            sh2rgb = eval_sh(pcs[0].active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_alpha(viewpoint_camera, pcs: List[GaussianModel], pipe,
           bg_color:torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
        Render the scene with alpha

        Background tensor (bg_color) must be on GPU!
    """

    ## merge sub point cloud to form full point clouds
    n = len(pcs)
    xyz = []
    features = []
    scaling = []
    rotation = []
    opacity = []
    cov3D_precomp_ = []
    point_numbers = []
    for ii in range(n):
        xyz.append(pcs[ii].get_xyz)
        point_numbers.append(xyz[ii].shape[0])
        features.append(pcs[ii].get_features)
        opacity.append(pcs[ii].get_opacity)
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        if pipe.compute_cov3D_python:
            cov3D_precomp_.append(pcs[ii].get_covariance(scaling_modifier))
        else:
            scaling.append(pcs[ii].get_scaling)
            rotation.append(pcs[ii].get_rotation)

    if override_color is not None:
        if isinstance(override_color, list) or isinstance(override_color, tuple):
            override_color = torch.cat(override_color, dim=0)

    scales = None
    rotations = None
    cov3D_precomp = None

    xyz = torch.cat(xyz, dim=0)
    features = torch.cat(features, dim=0)
    if pipe.compute_cov3D_python:
        cov3D_precomp = torch.cat(cov3D_precomp_, dim=0)
    else:
        scales = torch.cat(scaling, dim=0)
        rotations = torch.cat(rotation, dim=0)

    opacity = torch.cat(opacity, dim=0)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pcs[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    # opacity = opacity

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (pcs[0].max_sh_degree + 1) ** 2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pcs[0].active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, alpha0, radii = rasterizer( # DEBUG
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        alpha_point_id = 0
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha0": alpha0,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def get_depth(viewpoint_cameras, pcs: List[GaussianModel]):
    n = len(pcs)
    depth_map = []
    for ii in range(n):
        xyz = pcs[ii].get_xyz
        w2c_T = viewpoint_cameras.world_view_transform
        # Nx4
        homo_xyz = torch.cat([xyz,torch.ones_like(xyz[...,0:1])],dim=-1)
        c_xyzw = torch.matmul(homo_xyz, w2c_T)
        depth_c3 = c_xyzw[...,[2,2,2]]
        depth_map.append(depth_c3)
    return depth_map
