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

# Blend Shape

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation

from pytorch3d.io import load_obj

from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

#from .FLAME import FLAME, FLAMETex
from .FLAME import FLAME
from .rot import matrix_to_quaternion, quaternion_raw_multiply, quaternion_to_matrix
from diff_gaussian_rasterization import RotSH3_C3, CompositeBlendShape
#from diff_gaussian_rasterization import RotSH3_C3, RotSHBlendShape
from .transforms import normalize_transform
from .weight_volume import WeightVolume, WeightVolumeIndex
from .adamacc import AdamAcc

from enum import Enum

use_feature_expr_mask = True
use_rot_expr_mask = True
use_scale_expr_mask = True
use_opacity_expr_mask = True
use_xyz_expr_mask = True

use_trans_rot = True
use_trans_rot_sh = True

class DummyObj:
    def __init__(self):
        pass


class View(Enum):
    GROUND_TRUTH = 1
    COLOR_OVERLAY = 2
    SHAPE_OVERLAY = 4
    SHAPE = 8
    LANDMARKS = 16
    HEATMAP = 32
    DEPTH = 64
    MASK = 128

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def inv_softplus(x, beta=1., thres=20.):
            y = beta * x
            unsafe_mask = y > thres
            safe_y = torch.where(unsafe_mask, torch.full_like(y, thres), y)  # avoid nan
            z = torch.where(unsafe_mask, x, torch.log(torch.exp(safe_y) - 1 + 1e-12))
            return z

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):


        self.acc_dict = None
        #self.opt_id = 0

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            # new
            self._xyz_b,
            self._features_b,
            self._scaling_b,
            self._rotation_b,
            self._opacity_b,
            self.face_id,
            self.barycentric_coord,
            #
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        # new
        self._xyz_b,
        self._features_b,
        self._scaling_b,
        self._rotation_b,
        self._opacity_b,
        self.face_id,
        self.barycentric_coord,
        #
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        #if self.optimizer is not None:
        #    self.optimizer.load_state_dict(opt_dict)

    def sample_possion_points(self,v,f,N):
        import point_cloud_utils as pcu
        f_i, bc = pcu.sample_mesh_poisson_disk(v,f,N)
        print('expect point number %d, get %d' % (N, bc.shape[0]))
        return f_i, bc

    ########################

    def get_processed(self):
        #----
        _rotation = self._rotation
        init_rot = self.acc_dict["init_rot"]
        _rotation = _rotation + torch.sum(init_rot * self._mf, dim=-1)
        _rotation_b = self._rotation_b
        if use_rot_expr_mask:
            _rotation_b = _rotation_b * self.expr_mask_tensor.unsqueeze(1)
        _rotation = _rotation + torch.sum(_rotation_b * self._mf, dim=-1)
        processed_rotation = self.rotation_activation(_rotation)
        #----
        _features = self.processed_features
        init_sh = self.acc_dict["init_sh"]
        _features = _features + torch.sum(init_sh * self._mf, dim=-1)
        _features_b = self._features_b
        if use_feature_expr_mask:
            _features_b = _features_b * self.expr_mask_tensor.unsqueeze(1).unsqueeze(1)
        _features = _features + torch.sum(_features_b * self._mf, dim=-1)
        processed_features = _features
        return processed_rotation, processed_features

    def get_processed_acc(self):
        #----
        _rotation = self._rotation
        _rotation_b = self._rotation_b
        init_rot = self.acc_dict["init_rot"]
        _mask = self.expr_mask_tensor if use_rot_expr_mask else None
        _rotation = CompositeBlendShape(_rotation, _rotation_b, self._mf, init_rot, _mask)
        processed_rotation = self.rotation_activation(_rotation)
        #----
        _features = self.processed_features
        _features_b = self._features_b
        init_sh = self.acc_dict["init_sh"]
        _mask = self.expr_mask_tensor if use_feature_expr_mask else None
        processed_features = CompositeBlendShape(_features, _features_b, self._mf, init_sh, _mask)
        return processed_rotation, processed_features


    def run_blendshape(self, xyz, params):

        shape_params = params.shape
        trans_params = None
        rot_params = None
        neck_pose_params = None
        jaw_pose_params = params.jaw
        eye_pose_params = params.eyes
        expression_params = params.exp
        eyelid_params = params.eyelids

        batch_size = shape_params.shape[0]
        I = matrix_to_rotation_6d(torch.cat([torch.eye(3)[None]] * batch_size, dim=0).cuda())

        if trans_params is None:
            trans_params = torch.zeros(batch_size, 3).cuda()
        if rot_params is None:
            rot_params = I.clone()
        if neck_pose_params is None:
            neck_pose_params = I.clone()
        if jaw_pose_params is None:
            jaw_pose_params = I.clone()
        if eye_pose_params is None:
            eye_pose_params = torch.cat([I.clone()] * 2, dim=1)

        betas = torch.cat([shape_params, expression_params], dim=1) # 300 + 100
        full_pose = torch.cat([rot_params, neck_pose_params, jaw_pose_params, eye_pose_params], dim=1)

        #######################
        batch_size = 1
        device = betas.device
        dtype = torch.float32
        A = params.A
        A_diff = params.A_diff

        v_biased = xyz
        #n_gs_point = v_biased.shape[0]

        # Compute expr_mask_tensor
        self.expr_mask_tensor = self.acc_dict["expr_mask_tensor"]

        xyz_blendshape = self.acc_dict["init_xyz"] # Fx3x{50}
        v_shaped = v_biased + (xyz_blendshape * betas[...,300:350]).sum(-1) # Px3

        _xyz_b = self._xyz_b # Px3x50
        if use_xyz_expr_mask:
            _xyz_b = _xyz_b * self.expr_mask_tensor.unsqueeze(1)
        v_shaped = v_shaped + (_xyz_b * betas[...,300:350]).sum(-1)

        #processed_rotation, processed_features = self.get_processed()
        processed_rotation, processed_features = self.get_processed_acc()

        # Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = rotation_6d_to_matrix(full_pose.view(-1, 6)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])

        pose_f_tensor_ = self.acc_dict["pose_blendshape"]
        pose_f_tensor_ = pose_f_tensor_.view(36, -1)
        pose_offsets = torch.matmul(pose_feature, pose_f_tensor_).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped

        # fetch skinning weights
        W = self.acc_dict["W"]
        W = W.unsqueeze(dim=0).expand([batch_size, -1, -1])

        num_joints = W.shape[-1]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
        T_diff = torch.matmul(W, A_diff.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        ### eyelid
        eye_blendshape = self.acc_dict["eye_blendshape"]
        # eyelid_params 1x2
        verts = verts + (eye_blendshape * eyelid_params).sum(-1)

        vertices = verts
        vertices = vertices + trans_params.unsqueeze(dim=1)
        return vertices, T_diff, processed_rotation, processed_features


    def prepare_xyz(self, params , args):

        features_dc = self._features_dc
        features_rest = self._features_rest
        _features = torch.cat((features_dc, features_rest), dim=1)
        self.processed_features = _features

        vertices, T_diff, processed_rotation, processed_features = self.run_blendshape(
            self._xyz, params
        )
        self.processed_xyz = vertices[0]
        self.processed_rotation = processed_rotation
        self.processed_features = processed_features

        ## Rot gaussians and SH according to LBS (linear blend skinning) transform.
        if use_trans_rot or use_trans_rot_sh:
            rot_apply = normalize_transform(T_diff[0, :, :3, :3])

        if use_trans_rot:
            rot_q = matrix_to_quaternion(rot_apply)
            self.processed_rotation = quaternion_raw_multiply(rot_q, self.processed_rotation)
        if use_trans_rot_sh:
            self.processed_features = RotSH3_C3(rot_apply, self.processed_features)

    def prepare_merge(self, params):
        merged_f = params.exp[...,:self.n_expr_params]
        self._mf = merged_f[0]

    @property
    def get_scaling(self):
        _scaling = self._scaling
        _scaling_b = self._scaling_b
        if use_scale_expr_mask:
            _scaling_b = _scaling_b * self.expr_mask_tensor.unsqueeze(1)
        _scaling = _scaling + torch.sum(_scaling_b * self._mf, dim=-1)
        return self.scaling_activation(_scaling)

    @property
    def get_rotation(self):
        return self.processed_rotation

    @property
    def get_xyz(self):
        # WARNING! YOU SHOULD CALL PREPARE_XYZ BEFORE THIS FUNCTION
        return self.processed_xyz

    @property
    def get_features(self):
        return self.processed_features

    @property
    def get_opacity(self):
        _opacity = self._opacity
        _opacity_b = self._opacity_b
        if use_opacity_expr_mask:
            _opacity_b = _opacity_b * self.expr_mask_tensor.unsqueeze(1)
        _opacity = _opacity + torch.sum(_opacity_b * self._mf, dim=-1)
        return self.opacity_activation(_opacity)

    # def get_covariance(self, scaling_modifier = 1):
    #     return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def render_shape(self, vertices, faces=None, white=True):
        B = vertices.shape[0]
        V = vertices.shape[1]
        if faces is None:
            #faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
            faces = self.faces[None].repeat(B,1,1)
        if not white:
            verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=(1.0, 1.0, 1.0))

        fragments = self.mesh_rasterizer(meshes_world, cameras=self.cameras)
        rendering = self.debug_renderer.shader(fragments, meshes_world, cameras=self.cameras, blend_params=blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()
        return rendering[:, 0:3, :, :]

    def setup_renderer(self, args, img_size):
        raster_settings = RasterizationSettings(
            image_size=img_size,
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.lights = PointLights(
            device=args.data_device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )
        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=args.data_device, lights=self.lights)
        )

    def vis(self, params,  args, visualization): # []

        with torch.no_grad():
            self.cameras = params.cameras
            self.debug_renderer.rasterizer.raster_settings.image_size = params.image_size

            vertices, _, _ = self.flame(
                cameras=torch.inverse(params.cameras.R.to(args.data_device)),
                shape_params=params.shape,
                expression_params=params.exp,
                eye_pose_params=params.eyes, # 12
                jaw_pose_params=params.jaw, # 6
                eyelid_params=params.eyelids # 2
            )

            #albedos = self.albedos

            final_views = []
            for view in visualization:
                if view == View.COLOR_OVERLAY:
                    raise NotImplementedError
                if view == View.GROUND_TRUTH:
                    final_views.append(params.original_image.cpu().numpy())
                if view == View.MASK:
                    mask = params.mask.cpu().numpy()[...,None]
                    mask = np.repeat(mask,3,axis=-1)
                    final_views.append(mask)
                if view == View.SHAPE:
                    shape = self.render_shape(vertices, white=False)[0].cpu().numpy()
                    shape = np.transpose(shape,[1,2,0]) # CHW->HWC
                    final_views.append(shape)
                if view == View.LANDMARKS:
                    raise NotImplementedError
                if view == View.SHAPE_OVERLAY:
                    raise NotImplementedError
                if view == View.HEATMAP:
                    raise NotImplementedError

            return final_views

    def create_from_face(self, params, args, spatial_lr_scale: float, replace_v = True):

        self.spatial_lr_scale = spatial_lr_scale
        #fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        ## load canonical object
        mesh_file = args.flame_template_path #should be canonical mesh
        faces = load_obj(mesh_file) # Nx3 id int64
        canonical_verts = faces[0]
        self.faces = faces[1].verts_idx.to(args.data_device)

        ## load flame
        self.flame = FLAME(args).to(args.data_device)
        args.image_size = params.image_size
        args.actor = args.source_path
        #flametex = FLAMETex(args).to(args.data_device)
        #self.diff_renderer = Renderer(params.image_size, obj_fielname=mesh_file).to(self.data_device)

        ## load renderer
        self.setup_renderer(args, params.image_size)

        ## create init points

        if replace_v:
            from .transforms import get_rest_pose_vertices
            v = get_rest_pose_vertices(self,params,args).cpu().numpy()
        else:
            v = canonical_verts.cpu().numpy()
        f = faces[1].verts_idx.cpu().numpy()
        face_id, barycentric_coord = self.sample_possion_points(v,f, args.init_face_point_number)
        self.face_id = torch.from_numpy(face_id).to(args.data_device)
        self.barycentric_coord = torch.from_numpy(barycentric_coord).to(args.data_device)

        self.n_pose_params = 0
        self.n_expr_params = 50
        self.n_total_params = self.n_expr_params + self.n_pose_params

        with torch.no_grad():

            zeros_exp = torch.zeros_like(params.exp)
            zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
            zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
            zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

            vertices, _, _ = self.flame(
                cameras=torch.inverse(params.cameras.R.to(args.data_device)),
                shape_params=params.shape,
                expression_params=zeros_exp,
                eye_pose_params=zeros_eyes, # 12
                jaw_pose_params=zeros_jaw, # 6
                eyelid_params=zeros_eyelids # 2
            )

            #### Create grid for finding nearest triangle
            N_res1 = 256
            #N_res1 = 512
            path1 = os.path.join(args.source_path,"WeightID_%d.pt" % N_res1)
            if not os.path.exists(path1):
                print('gen weight volume id')
                path1 = None
            self.weight_volume_index = WeightVolumeIndex(
                path1,
                vertices[0],
                self.faces,
                N=N_res1
            )
            if path1 is None:
                print('done')
                path1 = os.path.join(args.source_path,"WeightID_%d.pt" % N_res1)
                self.weight_volume_index.dump(path1)

            #### Create grid for fetching LBS weights
            N_res2 = 256
            #N_res2 = 512
            path2 = os.path.join(args.source_path, "Weight5node_%d.pt" % N_res2)
            if not os.path.exists(path2):
                print('gen weight volume node')
                path2 = None
            self.weight_volume = WeightVolume(
                path2,
                vertices[0],
                self.flame.lbs_weights,
                self.faces,
                N=N_res2
            )
            if path2 is None:
                print('done')
                path2 = os.path.join(args.source_path, "Weight5node_%d.pt" % N_res2)
                self.weight_volume.dump(path2)

            ####

            vertices = vertices[0] # BxPx3 -> Px3

            p_face_id = self.faces[self.face_id]
            v_id_xyz = vertices[p_face_id]
            point_vertices = (v_id_xyz * self.barycentric_coord[..., None]).sum(-2)

            albedos = (torch.rand(point_vertices.shape[0],3)/255.).float().cuda()
            #fused_color = RGB2SH(albedos)
            fused_color = albedos
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features_b = torch.zeros((fused_color.shape[0], (self.max_sh_degree + 1) ** 2, 3, self.n_total_params)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0

            #dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            dist2 = torch.clamp_min(distCUDA2(point_vertices), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((point_vertices.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            scales_b = torch.zeros(list(scales.shape) + [self.n_total_params],dtype=torch.float,device="cuda")
            rots_b = torch.zeros(list(rots.shape) + [self.n_total_params],dtype=torch.float,device="cuda")

            #opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities = inverse_sigmoid(0.1 * torch.ones((point_vertices.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities_b = torch.zeros(list(opacities.shape)+[self.n_total_params],dtype=torch.float,device="cuda")

            xyz_b = torch.zeros((fused_color.shape[0], 3, self.n_total_params)).float().cuda()

        ## neutral model
        self._xyz = nn.Parameter(point_vertices.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._scaling.shape[0]), device="cuda")
        ## blendshapes
        self._xyz_b = nn.Parameter(xyz_b.requires_grad_(True))
        self._features_b = nn.Parameter(features_b.requires_grad_(True))
        self._scaling_b = nn.Parameter(scales_b.requires_grad_(True))
        self._rotation_b = nn.Parameter(rots_b.requires_grad_(True))
        self._opacity_b = nn.Parameter(opacities_b.requires_grad_(True))

        #flametex = None

    def training_setup(self, training_args, id = 0):
        # id = 0 full params
        # id = 1 neutral head params only
        # id = 2 blendshape diff only
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._scaling.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._scaling.shape[0], 1), device="cuda")

        l = []
        if id == 0 or id == 1:
            l += [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        if id == 0 or id == 2:
            weight_decay = training_args.weight_decay
            l += [
                {'params': [self._xyz_b], 'lr': 0.2 * 1.6e-6, "weight_decay": weight_decay, "name": "xyz_b"},
                {'params': [self._features_b], 'lr': 0.5 * training_args.feature_lr * training_args.basis_lr_decay, "weight_decay":weight_decay, "name": "f_b"},
                {'params': [self._opacity_b], 'lr': 0.001 * training_args.opacity_lr * training_args.basis_lr_decay, "weight_decay":weight_decay, "name": "opacity_b"},
                {'params': [self._scaling_b], 'lr': 0.1 * training_args.scaling_lr * training_args.basis_lr_decay, "weight_decay":weight_decay, "name": "scaling_b"},
                {'params': [self._rotation_b], 'lr': 0.1 * training_args.rotation_lr * training_args.basis_lr_decay, "weight_decay":weight_decay, "name": "rotation_b"}
            ]

        #self.opt_id = id
        #self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer = AdamAcc(l, lr=0.0, eps=1e-15) # ACC_FLAG
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def update_from_nn(self, update_acc_dict = False):

        v_biased = self._xyz
        # fetch skinning weights
        W = self.weight_volume(v_biased)
        W = W / torch.sum(W, dim=-1, keepdim=True)
        #W = W.unsqueeze(dim=0).expand([batch_size, -1, -1])

        # Compute expr_mask_tensor
        face_id, face_w = self.weight_volume_index(v_biased)  # Px3 -> Px8
        expr_mask_tensor_face = self.expr_mask_tensor_face[..., :50]  # Fx50
        face_id = face_id.long()
        tmp1 = expr_mask_tensor_face[face_id]  # Px8x50
        expr_mask_tensor = (tmp1 * face_w[..., None]).sum(1)

        if update_acc_dict:
            self.acc_dict.update({
                "expr_mask_tensor" : expr_mask_tensor.contiguous().detach(), # Px50?
                "W": W.contiguous().detach() # Px5
            })

        return expr_mask_tensor, W

    # Initialize difference of Gaussian blendshapes
    def compute_blendshape_init(self):
    #def extract_acc(self):
        # rot_tensor = Fx4x100
        # faceR_tensor = Fx100x3x3
        # facet_tensor = Fx100x3

        faceR_tensor = self.faceR_tensor
        facet_tensor = self.facet_tensor
        rot_tensor = self.rot_tensor
        pose_f_tensor = self.pose_f_tensor
        l_eyelid_offset = self.l_eyelid_offset
        r_eyelid_offset = self.r_eyelid_offset
        face_id = self.face_id # P

        v_biased = self._xyz

        ## Initialize rotation difference
        tmp1 = rot_tensor[face_id] # Px4x100
        tmp1 = tmp1[..., :50]
        tmp1_bak = tmp1 # Px4xN
        tmp1 = tmp1.permute(0, 2, 1)  # PxNx4
        _rotation = self._rotation  # Px4
        _rotation = torch.stack([_rotation] * 50, dim=-2)  # PxNx4
        _rot = quaternion_raw_multiply(tmp1, _rotation)
        _rot = _rot.permute(0, 2, 1)  # ->Px4xN
        init_rot = _rot

        ## Initialize SH difference
        # delta_sh = rot(qi,SH)-SH
        features_dc = self._features_dc
        features_rest = self._features_rest
        _features = torch.cat((features_dc, features_rest), dim=1) # Px16x3
        _rotation = self._rotation # Px4x100
        sh_blendshapes = []
        for ii in range(50):
            # Px4x100
            loc_rot = tmp1_bak[...,ii] # Px4
            # Get full rot
            loc_rot = loc_rot + torch.tensor([1.,0.,0.,0.],dtype=loc_rot.dtype,device=loc_rot.device)
            loc_mat = quaternion_to_matrix(loc_rot) # Px3x3
            rot_features = RotSH3_C3(loc_mat, _features)
            delta_  = rot_features - _features
            sh_blendshapes.append(delta_) # Px16x3
        init_sh = torch.stack(sh_blendshapes,dim=-1)

        ## Initialize pos. difference
        rd_faceR_tensor = faceR_tensor[:, :50, ...]  # Fx50x3x3
        rd_facet_tensor = facet_tensor[:, :50, ...]  # Fx50x3
        tmp1 = rd_faceR_tensor[face_id]  # Px50x3x3
        tmp2 = rd_facet_tensor[face_id]  # Px50x3
        v_biased_ = v_biased.unsqueeze(1).repeat(1, 50, 1)  # Px3 -> Px50x3
        tmp1 = tmp1.view(tmp1.shape[0] * 50, 3, 3)
        v_biased_ = v_biased_.view(tmp1.shape[0], 3, 1)
        tmp1 = torch.bmm(tmp1, v_biased_).squeeze(-1).view(tmp1.shape[0] // 50, 50, 3)
        xyz_blendshape = tmp1 + tmp2
        # Px50x3 -> Px3x50
        init_xyz = xyz_blendshape.permute(0, 2, 1)

        pose_f_tensor_ = pose_f_tensor[:,face_id] # 36xPx3
        pose_blendshape = pose_f_tensor_  # 36xPx3

        ## Initialize eyelid difference
        eye_offset = r_eyelid_offset[face_id] # Px3
        r_eye_offset = eye_offset
        eye_offset = l_eyelid_offset[face_id] # Px3
        l_eye_offset = eye_offset
        eye_blendshape = torch.stack([l_eye_offset, r_eye_offset], dim=-1)  # Px3x2

        expr_mask_tensor, W = self.update_from_nn()

        self.acc_dict = {
            "init_rot" : init_rot.contiguous().detach(), # Px4x50
            "init_sh" : init_sh.contiguous().detach(), # Px16x3x50
            "init_xyz" : init_xyz.contiguous().detach(), # Px3x50
            "pose_blendshape" : pose_blendshape.detach(), # 36xPx3
            "eye_blendshape" : eye_blendshape.detach(), # Px3x2
            # updatable
            "expr_mask_tensor" : expr_mask_tensor.contiguous().detach(), # Px50?
            "W": W.contiguous().detach() # Px5
        }

    def save_acc_dict(self, filename):
        acc_dict = self.acc_dict
        np_acc_dict = {}
        for k,v in acc_dict.items():
            np_acc_dict[k] = v.detach().cpu().numpy()
        np.save(filename,np_acc_dict)

    def load_acc_dict(self, filename):
        np_acc_dict = np.load(filename,allow_pickle=True).item()
        acc_dict = {}
        for k, v in np_acc_dict.items():
            acc_dict[k] = torch.from_numpy(v).cuda()
        self.acc_dict = acc_dict
        #self.flatten_rot = acc_dict["flatten_rot"]


    # dump for C++ viewer
    def save_npy_forviewer(self, path):

        mkdir_p(path)
        features_dc = self._features_dc
        features_reset = self._features_rest
        _xyz = self._xyz
        _features = torch.cat((features_dc, features_reset),dim=1)
        _scaling = self._scaling
        _rotation = self._rotation
        _opacity = self._opacity

        assert(self.acc_dict is not None)
        extract_params = self.acc_dict

        _xyz = _xyz.cpu().detach().numpy()
        _features = _features.cpu().detach().numpy()
        n_points = _xyz.shape[0]

        # create valid mask
        sel_id = np.ones([n_points],dtype=np.bool)
        set1 = np.nonzero(np.isnan(_xyz).reshape(n_points,-1).max(-1))
        set2 = np.nonzero(np.isnan(_features).reshape(n_points,-1).max(-1))
        # if len(set1) > 0:
        #     print('find %d nan in set1' % len(set1))
        # if len(set2) > 0:
        #     print('find %d nan in set2' % len(set2))
        for id in set1:
            sel_id[id] = False
        for id in set2:
            sel_id[id] = False

        def save_(filename, o):
            np.save(filename,o[sel_id])

        save_(os.path.join(path, "pos.npy"),_xyz)
        save_(os.path.join(path, "shs.npy"),_features)
        save_(os.path.join(path, "scale.npy"), _scaling.cpu().detach().numpy())  # 71600x3
        save_(os.path.join(path, "rot.npy"), _rotation.cpu().detach().numpy())  # 71600x4
        save_(os.path.join(path, "opacity.npy"), _opacity.cpu().detach().numpy())  # 71600x1

        #v_biased = self._xyz
        # fetch skinning weights
        W = self.acc_dict["W"]

        # Compute expr_mask_tensor
        expr_mask_tensor = self.acc_dict["expr_mask_tensor"]

        init_rot = extract_params["init_rot"]
        init_sh = extract_params["init_sh"]
        init_xyz = extract_params["init_xyz"]
        pose_blendshape = extract_params["pose_blendshape"]
        eye_blendshape = extract_params["eye_blendshape"]

        _features_b = self._features_b
        if use_feature_expr_mask:
            _features_b = _features_b * expr_mask_tensor.unsqueeze(1).unsqueeze(1)
        _scaling_b = self._scaling_b
        if use_scale_expr_mask:
            _scaling_b = _scaling_b * expr_mask_tensor.unsqueeze(1)
        _rotation_b = self._rotation_b
        if use_rot_expr_mask:
            _rotation_b = _rotation_b * expr_mask_tensor.unsqueeze(1)
        _opacity_b = self._opacity_b
        if use_opacity_expr_mask:
            _opacity_b = _opacity_b * expr_mask_tensor.unsqueeze(1)

        _xyz_b = self._xyz_b
        if use_xyz_expr_mask:
            _xyz_b = _xyz_b * expr_mask_tensor.unsqueeze(1)
        xyz_blendshape = _xyz_b + init_xyz

        _rotation_b = _rotation_b + init_rot
        _features_b = _features_b + init_sh

        save_(os.path.join(path,"xyz_t.npy"),xyz_blendshape.cpu().detach().numpy()) # Px3x100
        save_(os.path.join(path,"shs_t.npy"),_features_b.cpu().detach().numpy()) # PxLx3x100
        save_(os.path.join(path,"scale_t.npy"),_scaling_b.cpu().detach().numpy()) # Px3x100
        save_(os.path.join(path,"rot_t.npy"),_rotation_b.cpu().detach().numpy()) # Px4x100
        save_(os.path.join(path,"opacity_t.npy"),_opacity_b.cpu().detach().numpy()) # Px1x100

        np.save(os.path.join(path, "pos_t.npy"), pose_blendshape.cpu().detach().numpy()[:,sel_id])  # 36xPx3
        save_(os.path.join(path, "W.npy"), W.cpu().detach().numpy()) # Px5
        save_(os.path.join(path, "eyelid.npy"), eye_blendshape.cpu().detach().numpy()) # Px3x2


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        if "xyz" in optimizable_tensors:
            self._xyz = optimizable_tensors["xyz"]
        else:
            tmp = self._xyz[valid_points_mask].detach()
            self._xyz = nn.Parameter(tmp.requires_grad_(True))
        #
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        else:
            tmp = self._features_dc[valid_points_mask].detach()
            self._features_dc = nn.Parameter(tmp.requires_grad_(True))
        #
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]
        else:
            tmp = self._features_rest[valid_points_mask].detach()
            self._features_rest = nn.Parameter(tmp.requires_grad_(True))
        #
        if "opacity" in optimizable_tensors:
            self._opacity = optimizable_tensors["opacity"]
        else:
            tmp = self._opacity[valid_points_mask].detach()
            self._opacity = nn.Parameter(tmp.requires_grad_(True))
        #
        if "scaling" in optimizable_tensors:
            self._scaling = optimizable_tensors["scaling"]
        else:
            tmp = self._scaling[valid_points_mask].detach()
            self._scaling = nn.Parameter(tmp.requires_grad_(True))
        #
        if "rotation" in optimizable_tensors:
            self._rotation = optimizable_tensors["rotation"]
        else:
            tmp = self._rotation[valid_points_mask].detach()
            self._rotation = nn.Parameter(tmp.requires_grad_(True))

        if "xyz_b" in optimizable_tensors:
            self._xyz_b = optimizable_tensors["xyz_b"]
        else:
            tmp = self._xyz_b[valid_points_mask].detach()
            self._xyz_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "f_b" in optimizable_tensors:
            self._features_b = optimizable_tensors["f_b"]
        else:
            tmp = self._features_b[valid_points_mask].detach()
            self._features_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "opacity_b" in optimizable_tensors:
            self._opacity_b = optimizable_tensors["opacity_b"]
        else:
            tmp = self._opacity_b[valid_points_mask].detach()
            self._opacity_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "scaling_b" in optimizable_tensors:
            self._scaling_b = optimizable_tensors["scaling_b"]
        else:
            tmp = self._scaling_b[valid_points_mask].detach()
            self._scaling_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "rotation_b" in optimizable_tensors:
            self._rotation_b = optimizable_tensors["rotation_b"]
        else:
            tmp = self._rotation_b[valid_points_mask].detach()
            self._rotation_b = nn.Parameter(tmp.requires_grad_(True))

        if self.acc_dict is not None:
            self.acc_dict["init_rot"] = self.acc_dict["init_rot"][valid_points_mask].detach()
            self.acc_dict["init_sh"] = self.acc_dict["init_sh"][valid_points_mask].detach()
            self.acc_dict["init_xyz"] = self.acc_dict["init_xyz"][valid_points_mask].detach()
            self.acc_dict["pose_blendshape"] = self.acc_dict["pose_blendshape"][:, valid_points_mask].detach()
            self.acc_dict["eye_blendshape"] = self.acc_dict["eye_blendshape"][valid_points_mask].detach()
            self.acc_dict["expr_mask_tensor"] = self.acc_dict["expr_mask_tensor"][valid_points_mask].detach()
            self.acc_dict["W"] = self.acc_dict["W"][valid_points_mask].detach()

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                              new_xyz_b,new_features_b, new_opacity_b, new_scaling_b, new_rotation_b, new_acc_dict):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             #
             "xyz_b": new_xyz_b,
             "f_b": new_features_b,
             "opacity_b": new_opacity_b,
             "scaling_b": new_scaling_b,
             "rotation_b": new_rotation_b
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        #
        if "xyz" in optimizable_tensors:
            self._xyz = optimizable_tensors["xyz"]
        else:
            tmp = torch.cat([self._xyz, new_xyz],dim=0).detach()
            self._xyz = nn.Parameter(tmp.requires_grad_(True))
        #
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        else:
            tmp = torch.cat([self._features_dc, new_features_dc],dim=0).detach()
            self._features_dc = nn.Parameter(tmp.requires_grad_(True))
        #
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]
        else:
            tmp = torch.cat([self._features_rest, new_features_rest],dim=0).detach()
            self._features_rest = nn.Parameter(tmp.requires_grad_(True))
        #
        if "opacity" in optimizable_tensors:
            self._opacity = optimizable_tensors["opacity"]
        else:
            tmp = torch.cat([self._opacity, new_opacities], dim = 0).detach()
            self._opacity = nn.Parameter(tmp.requires_grad_(True))
        #
        if "scaling" in optimizable_tensors:
            self._scaling = optimizable_tensors["scaling"]
        else:
            tmp = torch.cat([self._scaling, new_scaling], dim = 0).detach()
            self._scaling = nn.Parameter(tmp.requires_grad_(True))
        #
        if "rotation" in optimizable_tensors:
            self._rotation = optimizable_tensors["rotation"]
        else:
            tmp = torch.cat([self._rotation, new_rotation], dim = 0).detach()
            self._rotation = nn.Parameter(tmp.requires_grad_(True))

        #
        if "xyz_b" in optimizable_tensors:
            self._xyz_b = optimizable_tensors["xyz_b"]
        else:
            tmp = torch.cat([self._xyz_b,new_xyz_b],dim=0).detach()
            self._xyz_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "f_b" in optimizable_tensors:
            self._features_b = optimizable_tensors["f_b"]
        else:
            tmp = torch.cat([self._features_b,new_features_b],dim=0).detach()
            self._features_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "opacity_b" in optimizable_tensors:
            self._opacity_b = optimizable_tensors["opacity_b"]
        else:
            tmp = torch.cat([self._opacity_b, new_opacity_b],dim=0).detach()
            self._opacity_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "scaling_b" in optimizable_tensors:
            self._scaling_b = optimizable_tensors["scaling_b"]
        else:
            tmp = torch.cat([self._scaling_b, new_scaling_b],dim=0).detach()
            self._scaling_b = nn.Parameter(tmp.requires_grad_(True))
        #
        if "rotation_b" in optimizable_tensors:
            self._rotation_b = optimizable_tensors["rotation_b"]
        else:
            tmp = torch.cat([self._rotation_b, new_rotation_b],dim=0).detach()
            self._rotation_b = nn.Parameter(tmp.requires_grad_(True))

        self.xyz_gradient_accum = torch.zeros((self._scaling.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._scaling.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._scaling.shape[0]), device="cuda")

        with torch.no_grad():
            if self.acc_dict is not None:
                self.acc_dict["init_rot"] = torch.cat([self.acc_dict["init_rot"], new_acc_dict["init_rot"]],dim=0)
                self.acc_dict["init_sh"] = torch.cat([self.acc_dict["init_sh"], new_acc_dict["init_sh"]], dim=0)
                self.acc_dict["init_xyz"] = torch.cat([self.acc_dict["init_xyz"], new_acc_dict["init_xyz"]], dim=0)
                self.acc_dict["pose_blendshape"] = torch.cat([self.acc_dict["pose_blendshape"], new_acc_dict["pose_blendshape"]],dim=1)
                self.acc_dict["eye_blendshape"] = torch.cat([self.acc_dict["eye_blendshape"], new_acc_dict["eye_blendshape"]],dim=0)
                self.acc_dict["expr_mask_tensor"] = torch.cat([self.acc_dict["expr_mask_tensor"], new_acc_dict["expr_mask_tensor"]],dim=0)
                self.acc_dict["W"] = torch.cat([self.acc_dict["W"], new_acc_dict["W"]],dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):

        _scaling = self.scaling_activation(self._scaling)
        n_init_points = self._scaling.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        #selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(_scaling, dim=1).values > self.percent_dense * scene_extent)

        #stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = _scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        #new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        #new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_scaling = self.scaling_inverse_activation(_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_xyz_b = self._xyz_b[selected_pts_mask].repeat(N,1,1)
        new_features_b = self._features_b[selected_pts_mask].repeat(N,1,1,1)
        new_opacity_b = self._opacity_b[selected_pts_mask].repeat(N,1,1)
        new_scaling_b = self._scaling_b[selected_pts_mask].repeat(N,1,1)
        new_rotation_b = self._rotation_b[selected_pts_mask].repeat(N,1,1)

        with torch.no_grad():
            new_acc_dict = None
            if self.acc_dict is not None:
                new_acc_dict = {
                    "init_rot": self.acc_dict["init_rot"][selected_pts_mask].repeat(N,1,1),
                    "init_sh": self.acc_dict["init_sh"][selected_pts_mask].repeat(N,1,1,1),
                    "init_xyz": self.acc_dict["init_xyz"][selected_pts_mask].repeat(N,1,1),
                    "pose_blendshape": self.acc_dict["pose_blendshape"][:, selected_pts_mask].repeat(1,N,1),
                    "eye_blendshape": self.acc_dict["eye_blendshape"][selected_pts_mask].repeat(N,1,1),
                    "expr_mask_tensor": self.acc_dict["expr_mask_tensor"][selected_pts_mask].repeat(N,1),
                    "W": self.acc_dict["W"][selected_pts_mask].repeat(N,1)
                }

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                    new_xyz_b,new_features_b, new_opacity_b, new_scaling_b, new_rotation_b, new_acc_dict)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        _scaling = self.scaling_activation(self._scaling)
        #selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(_scaling,  dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_xyz_b = self._xyz_b[selected_pts_mask]
        new_features_b = self._features_b[selected_pts_mask]
        new_opacity_b = self._opacity_b[selected_pts_mask]
        new_scaling_b = self._scaling_b[selected_pts_mask]
        new_rotation_b = self._rotation_b[selected_pts_mask]

        with torch.no_grad():
            new_acc_dict = None
            if self.acc_dict is not None:
                new_acc_dict = {
                    "init_rot": self.acc_dict["init_rot"][selected_pts_mask],
                    "init_sh": self.acc_dict["init_sh"][selected_pts_mask],
                    "init_xyz": self.acc_dict["init_xyz"][selected_pts_mask],
                    "pose_blendshape": self.acc_dict["pose_blendshape"][:, selected_pts_mask],
                    "eye_blendshape": self.acc_dict["eye_blendshape"][selected_pts_mask],
                    "expr_mask_tensor": self.acc_dict["expr_mask_tensor"][selected_pts_mask],
                    "W": self.acc_dict["W"][selected_pts_mask]
                }

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                   new_xyz_b,new_features_b, new_opacity_b, new_scaling_b, new_rotation_b, new_acc_dict)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        _scaling = self.scaling_activation(self._scaling)

        #prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = (self.opacity_activation(self._opacity) < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            #big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            big_points_ws = _scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1





