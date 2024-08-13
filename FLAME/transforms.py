import numpy as np
import torch
import os
#from torch import nn
import torch.nn.functional as F
from .rot import matrix_to_quaternion
from tqdm import tqdm

from diff_gaussian_rasterization import InvMat3x3

#from torch.autograd import Function


def create_local_frame(face_vertex):
    # vertex Fx3(id)x3(xyz)
    a1 = face_vertex[:,1] - face_vertex[:,0] # v2 - v1
    a2 = face_vertex[:,2] - face_vertex[:,0] # v3 - v1
    b1 = F.normalize(a1,dim=-1)
    b2 = a2 - (b1 * a2).sum(-1,keepdim=True) * b1
    b2 = F.normalize(b2,dim=-1)
    b3 = torch.cross(b1,b2,dim=-1)
    return torch.stack([b1,b2,b3],dim=-2)


# Extract rotation matrix from affine matrix
# Shoemake and Duff 1992.
# https://blog.csdn.net/qq_39300235/article/details/105790743
def normalize_transform(transform):
    R = transform
    for ii in range(3):
        # NOTE: pytorch's implementation of inversion for batchified 3x3 matrix is slow
        R_u = 0.5 * (R + InvMat3x3(R.permute(0,2,1)))
        R = R_u
    return R


def create_transform(face_vertex_src, face_vertex_tar):
    # Deformation Transfer for Triangle Meshes
    # https://people.csail.mit.edu/sumner/research/deftransfer/
    center = (face_vertex_src[:,0] + face_vertex_src[:,1] + face_vertex_src[:,2])/3.
    d1 = face_vertex_src[:,1] - face_vertex_src[:,0]
    d2 = face_vertex_src[:,2] - face_vertex_src[:,0]
    d3 = torch.cross(d1,d2,dim=-1)
    #face_vertex_perpendicular = d3 / torch.sqrt((d3 * d3).sum(-1,keepdim=True))
    face_vertex_perpendicular = d3 / torch.sqrt(torch.linalg.norm(d3,dim=-1,keepdim=True))
    d3 = center + face_vertex_perpendicular - face_vertex_src[:,0]

    center_ = (face_vertex_tar[:,0] + face_vertex_tar[:,1] + face_vertex_tar[:,2])/3.
    d1_ = face_vertex_tar[:,1] - face_vertex_tar[:,0]
    d2_ = face_vertex_tar[:,2] - face_vertex_tar[:,0]
    d3_ = torch.cross(d1_,d2_,dim=-1)
    #face_vertex_perpendicular_ = d3_ / torch.sqrt((d3_ * d3_).sum(-1,keepdim=True))
    face_vertex_perpendicular_ = d3_ / torch.sqrt(torch.linalg.norm(d3_,dim=-1,keepdim=True))
    d3_ = center_ + face_vertex_perpendicular_ - face_vertex_tar[:,0]

    V = torch.stack([d1,d2,d3],dim=-1)
    V_ = torch.stack([d1_,d2_,d3_],dim=-1)
    Q = torch.bmm(V_,torch.inverse(V))
    b = face_vertex_tar[:,0] - torch.bmm(Q,face_vertex_src[:,0].unsqueeze(-1)).squeeze(-1)
    return Q,b

def get_all_touched_index(f,face_id_list):
    touched_vertex_idx = []
    for i_face in face_id_list:
        touched_vertex_idx.append(f[i_face][0])
        touched_vertex_idx.append(f[i_face][1])
        touched_vertex_idx.append(f[i_face][2])
    touched_vertex_idx = list(set(touched_vertex_idx))  # remove dumplicate
    return touched_vertex_idx

@torch.no_grad()
def get_eyelid_tensor(face_gaussians, dummy_frame, mask_func, args):

    from .rigid_transform_3D import rigid_transform_3D

    back_head_face_idx = np.loadtxt(args.back_head_file)
    back_head_face_idx = np.asarray(back_head_face_idx,dtype=np.int32)

    flame = face_gaussians.flame
    face_id = face_gaussians.face_id.cpu().detach().numpy()
    faces = face_gaussians.faces.cpu().detach().numpy()
    barycentric_coord = face_gaussians.barycentric_coord.cpu().detach().numpy()
    p_face_id = faces[face_id]

    back_head_vertex_idx = get_all_touched_index(faces, back_head_face_idx)

    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    vertices_neutral, _, _ = flame(
        cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params=dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,  # 12
        jaw_pose_params=zeros_jaw,  # 6
        eyelid_params=zeros_eyelids  # 2
    )
    vertices_neutral = vertices_neutral[0].cpu().detach().numpy()

    transfer_tensor = []
    for eyelid_id in range(2):
        zeros_eyelids = torch.zeros_like(zeros_eyelids)
        zeros_eyelids[0,eyelid_id] = 1.
        vertices_eyelid,_,_ = flame(
            cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
            shape_params=dummy_frame.shape,
            expression_params=zeros_exp,
            eye_pose_params=zeros_eyes,  # 12
            jaw_pose_params=zeros_jaw,  # 6
            eyelid_params=zeros_eyelids  # 2
        )

        vertices_eyelid = vertices_eyelid[0].cpu().detach().numpy()
        points_eyelid = vertices_eyelid[back_head_vertex_idx]
        points_neutral = vertices_neutral[back_head_vertex_idx]
        R, t = rigid_transform_3D(points_eyelid.T, points_neutral.T) # exp ->neutral
        points_eyelid_transfer = vertices_eyelid @ R.T + t[...,0]
        mesh_vertex_offset = points_eyelid_transfer - vertices_neutral

        # interpolate to triangles
        v_id_offset = mesh_vertex_offset[p_face_id]
        point_offset = (v_id_offset * barycentric_coord[..., None]).sum(-2)

        transfer_tensor.append(point_offset)

    transfer_tensor = np.stack(transfer_tensor,axis=-1) # Px3x2(K)
    transfer_tensor = torch.from_numpy(transfer_tensor).to(args.data_device)
    eyelid_mask_tensor = mask_func(transfer_tensor, args)
    face_gaussians.eyelid_mask_tensor = eyelid_mask_tensor.detach()
    del transfer_tensor

@torch.no_grad()
def get_jaw_tensor(face_gaussians, args, clip_thresh = 0.1):
    flame = face_gaussians.flame
    jaw_weights = flame.lbs_weights[:,2]

    # apply a function here
    max_v = jaw_weights.max()
    min_v = max_v * clip_thresh
    norm_w = torch.clip(jaw_weights - min_v,0.)/ (max_v - min_v)

    face_id = face_gaussians.face_id
    faces = face_gaussians.faces
    barycentric_coord = face_gaussians.barycentric_coord
    p_face_id = faces[face_id] # Fx3
    v_id_offset = norm_w[p_face_id] # Fx3
    #print('v_id_offset',v_id_offset.shape)
    #print('barycentric_coord',barycentric_coord.shape)
    #exit(-1)
    point_offset = (v_id_offset * barycentric_coord).sum(-1) # Fx3 -> F
    face_gaussians.jaw_mask_tensor = point_offset

@torch.no_grad()
def get_pose_tensor(model, args):

    flame = model.flame
    faces = model.faces

    pose_v_tensor = flame.posedirs # 36x(Vx3)
    pose_v_tensor = pose_v_tensor.view(pose_v_tensor.shape[0],-1,3) # 36xVx3
    pose_f_tensor = pose_v_tensor[:,faces] # 36x{Fx3}x3
    pose_f_tensor = pose_f_tensor.sum(2) # 36xFx3
    model.pose_f_tensor = pose_f_tensor

    l_eyelid_offset = flame.l_eyelid[0] # 1x5023x3->5023x3
    l_eyelid_offset = l_eyelid_offset[faces] # {Fx3}x3
    l_eyelid_offset = l_eyelid_offset.mean(1) # Fx3

    r_eyelid_offset = flame.r_eyelid[0] # 1x5023x3->5023x3
    r_eyelid_offset = r_eyelid_offset[faces] # {Fx3}x3
    r_eyelid_offset = r_eyelid_offset.mean(1) # Fx3
    model.l_eyelid_offset = l_eyelid_offset
    model.r_eyelid_offset = r_eyelid_offset

def get_expr_consistency_face(face_gaussians, dummy_frame, mask_func, args, ignore_neck=False):
    from .rigid_transform_3D import rigid_transform_3D

    if ignore_neck:
        tmp_name = os.path.dirname(args.back_head_file)
        tmp_name = os.path.join(tmp_name,"neck_exclude_vertex.txt")
        neck_exclude_vertex = np.loadtxt(tmp_name,dtype=np.int32)

    back_head_face_idx = np.loadtxt(args.back_head_file)
    back_head_face_idx = np.asarray(back_head_face_idx, dtype=np.int32)

    flame = face_gaussians.flame
    #face_id = face_gaussians.face_id.cpu().detach().numpy()
    faces = face_gaussians.faces.cpu().detach().numpy()
    #barycentric_coord = face_gaussians.barycentric_coord.cpu().detach().numpy()

    back_head_vertex_idx = get_all_touched_index(faces, back_head_face_idx)

    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    vertices_neutral, _, _ = flame(
        cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params=dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,  # 12
        jaw_pose_params=zeros_jaw,  # 6
        eyelid_params=zeros_eyelids  # 2
    )
    vertices_neutral = vertices_neutral[0].cpu().detach().numpy()

    transfer_tensor_face = []
    for exp_id in range(100):
        zeros_exp = torch.zeros_like(dummy_frame.exp)
        zeros_exp[0, exp_id] = 1.
        vertices_exp, _, _ = flame(
            cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
            shape_params=dummy_frame.shape,
            expression_params=zeros_exp,
            eye_pose_params=zeros_eyes,  # 12
            jaw_pose_params=zeros_jaw,  # 6
            eyelid_params=zeros_eyelids  # 2
        )

        vertices_exp = vertices_exp[0].cpu().detach().numpy()
        points_exp = vertices_exp[back_head_vertex_idx]
        points_neutral = vertices_neutral[back_head_vertex_idx]
        R, t = rigid_transform_3D(points_exp.T, points_neutral.T)  # exp ->neutral
        points_exp_transfer = vertices_exp @ R.T + t[..., 0]
        mesh_vertex_offset = points_exp_transfer - vertices_neutral
        if ignore_neck:
            mesh_vertex_offset[neck_exclude_vertex] = 0.
        # Compute face offset with vertex offset
        face_offset = mesh_vertex_offset[faces].mean(1) # {Fx3}x3
        transfer_tensor_face.append(face_offset)

    transfer_tensor_face = np.stack(transfer_tensor_face, axis=-1)  # Fx3x100
    transfer_tensor_face = torch.from_numpy(transfer_tensor_face).to(args.data_device)
    expr_mask_tensor_face = mask_func(transfer_tensor_face, args)
    face_gaussians.expr_mask_tensor_face = expr_mask_tensor_face.detach()
    del transfer_tensor_face

@torch.no_grad()
def get_expr_tensor(face_gaussians, dummy_frame, mask_func, args):

    from .rigid_transform_3D import rigid_transform_3D

    back_head_face_idx = np.loadtxt(args.back_head_file)
    back_head_face_idx = np.asarray(back_head_face_idx,dtype=np.int32)

    flame = face_gaussians.flame
    face_id = face_gaussians.face_id.cpu().detach().numpy()
    faces = face_gaussians.faces.cpu().detach().numpy()
    barycentric_coord = face_gaussians.barycentric_coord.cpu().detach().numpy()
    # faces = flame.faces.cpu().detach().numpy() # should not use flame faces ...
    p_face_id = faces[face_id]

    back_head_vertex_idx = get_all_touched_index(faces,back_head_face_idx)

    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    vertices_neutral, _, _ = flame(
        cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params=dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,  # 12
        jaw_pose_params=zeros_jaw,  # 6
        eyelid_params=zeros_eyelids  # 2
    )
    vertices_neutral =  vertices_neutral[0].cpu().detach().numpy()

    transfer_tensor = []
    for exp_id in range(100):

        zeros_exp = torch.zeros_like(dummy_frame.exp)
        zeros_exp[0,exp_id] = 1.
        vertices_exp, _, _ = flame(
            cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
            shape_params=dummy_frame.shape,
            expression_params=zeros_exp,
            eye_pose_params=zeros_eyes,  # 12
            jaw_pose_params=zeros_jaw,  # 6
            eyelid_params=zeros_eyelids  # 2
        )

        vertices_exp = vertices_exp[0].cpu().detach().numpy()
        points_exp = vertices_exp[back_head_vertex_idx]
        points_neutral = vertices_neutral[back_head_vertex_idx]
        R, t = rigid_transform_3D(points_exp.T, points_neutral.T) # exp ->neutral
        points_exp_transfer = vertices_exp @ R.T + t[...,0]
        mesh_vertex_offset = points_exp_transfer - vertices_neutral
        # interpolate to gs points
        v_id_offset = mesh_vertex_offset[p_face_id]
        point_offset = (v_id_offset * barycentric_coord[..., None]).sum(-2)

        transfer_tensor.append(point_offset)

    transfer_tensor = np.stack(transfer_tensor,axis=-1) # Px3x100
    print('transfer_tensor.shape',transfer_tensor.shape)
    transfer_tensor = torch.from_numpy(transfer_tensor).to(args.data_device)
    expr_mask_tensor = mask_func(transfer_tensor,args)
    print('expr_mask_tensor.shape', expr_mask_tensor.shape)
    face_gaussians.expr_mask_tensor = expr_mask_tensor.detach()
    del transfer_tensor

@torch.no_grad()
def get_expr_rot(face_gaussians, dummy_frame, args, light=False):

    flame = face_gaussians.flame
    faces = face_gaussians.faces

    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    vertices_neutral, _, _ = flame(
        cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params=dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,  # 12
        jaw_pose_params=zeros_jaw,  # 6
        eyelid_params=zeros_eyelids  # 2
    )
    vertices_neutral = vertices_neutral[0]
    v_id_xyz0 = vertices_neutral[faces]

    rot_list = []
    face_offset_list = []
    faceR_list = []
    facet_list = []
    for exp_id in range(100):
        zeros_exp = torch.zeros_like(dummy_frame.exp)
        zeros_exp[0, exp_id] = 1.
        vertices_exp, _, _ = flame(
            cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
            shape_params=dummy_frame.shape,
            expression_params=zeros_exp,
            eye_pose_params=zeros_eyes,  # 12
            jaw_pose_params=zeros_jaw,  # 6
            eyelid_params=zeros_eyelids  # 2
        )
        vertices_exp = vertices_exp[0] # Nx3
        v_id_xyz = vertices_exp[faces]
        faceR, facet = create_transform(v_id_xyz0, v_id_xyz) # neutral -> exp
        faceR = faceR.to(args.data_device)
        #facet = facet.to(args.data_device)
        faceNR = normalize_transform(faceR)
        iden_q = torch.tensor([1.,0.,0.,0.],device=faceR.device)
        rot_q = matrix_to_quaternion(faceNR)
        diff_q = rot_q - iden_q # Fx4
        rot_list.append(diff_q)
        #
        face_offset = v_id_xyz - v_id_xyz0 # {Fx3}x3
        face_offset = face_offset.mean(1)
        face_offset_list.append(face_offset) # Fx3
        #
        faceR_list.append(faceR) # Fx3x3
        facet_list.append(facet) # Fx3

    rot_tensor = torch.stack(rot_list,dim=-1) # Fx4x100
    if light: # tensor for each triangle face
        face_gaussians.rot_tensor = rot_tensor
    else: # tensor for each Gaussian
        face_id = face_gaussians.face_id
        face_gaussians.rot_tensor = rot_tensor[face_id] # Px4x100

    face_offset_tensor = torch.stack(face_offset_list,dim=-1) # Fx3x100
    face_gaussians.face_offset_tensor = face_offset_tensor

    faceR_tensor = torch.stack(faceR_list,dim=1) # Fx100x3x3
    facet_tensor = torch.stack(facet_list,dim=1) # Fx100x3

    face_gaussians.faceR_tensor = faceR_tensor - torch.eye(3,device=faceR_tensor.device)
    face_gaussians.facet_tensor = facet_tensor
    # face_gaussians.facet_tensor = facet_tensor


@torch.no_grad()
def get_rest_pose_vertices(face_gaussians, dummy_frame, args):

    flame = face_gaussians.flame
    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    vertices0, _, _ = flame(
        cameras = torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params = dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,
        jaw_pose_params=zeros_jaw,
        eyelid_params=zeros_eyelids
    )
    vertices0 = vertices0[0] # BxPx3 -> Px3
    return vertices0


class MouthObj:
    def __init__(self,mouth_file):
        import point_cloud_utils as pcu
        v,f = pcu.load_mesh_vf(mouth_file)# fp64,132x3
        v = np.asarray(v,dtype=np.float32)# int64,44x3

        self.n_base_face = 0

        self.vertex = v
        self.face = f
        self.point_face_id = None
        self.barycentric_coord = None

        self.ts_vertex = None
        self.ts_face = None
        self.ts_point_face_id = None
        self.ts_barycentric_coord = None

    def offset_obj(self):
        mid = (self.upper_align_point + self.bottom_align_point) * 0.5
        v = self.vertex
        v = v + mid.cpu().detach().numpy() + np.array([0.,0.,-0.01],dtype=np.float32)
        self.vertex = v

    def append(self, v, f):
        n_vertex = self.vertex.shape[0]
        n_face = self.face.shape[0]
        self.vertex = np.concatenate([self.vertex,v],axis=0)
        f = f + n_vertex # add vertex offset
        self.face = np.concatenate([self.face,f],axis=0)
        self.n_base_face = n_face

    def sample(self, n_expect):
        import point_cloud_utils as pcu
        f_i, bc = pcu.sample_mesh_poisson_disk(self.vertex,self.face,n_expect)
        print('MouseObj, expect point number %d, get %d' % (n_expect,bc.shape[0]))
        self.point_face_id = f_i
        self.barycentric_coord = bc

    def to_tensor(self,device):
        self.ts_vertex = torch.from_numpy(self.vertex).to(device)
        self.ts_face = torch.from_numpy(self.face).to(device)
        self.ts_point_face_id = torch.from_numpy(self.point_face_id).to(device)
        self.ts_barycentric_coord = torch.from_numpy(self.barycentric_coord).to(device)

    def get_points(self,offset_upper, offset_bottom):
        p_face_id = self.ts_face[self.ts_point_face_id]
        v_id_xyz = self.ts_vertex[p_face_id]
        point_vertices = (v_id_xyz * self.ts_barycentric_coord[...,None]).sum(-2)
        fix_points = point_vertices[self.ts_point_face_id < self.n_base_face]
        upper_points = point_vertices[torch.logical_and(self.ts_point_face_id >= self.n_base_face, self.ts_point_face_id < self.n_base_face + 2)]
        bottom_points = point_vertices[self.ts_point_face_id >= self.n_base_face + 2]
        upper_points = upper_points + offset_upper
        bottom_points = bottom_points + offset_bottom
        #print('debug get points number', fix_points.shape[0], upper_points.shape[0], bottom_points.shape[0])
        return torch.concat([fix_points, upper_points, bottom_points],dim=0), \
               (fix_points.shape[0], upper_points.shape[0], bottom_points.shape[0])

@torch.no_grad()
def rigid_transfer_for_mouth2(dataset, model, args):

    flame = model.flame
    faces = model.faces
    dummy_frame = dataset.output_list[0]

    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    jawTrans0 = flame.get_jaw_transfer(
        cameras = torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params=dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,
        jaw_pose_params=zeros_jaw,
        eyelid_params=zeros_eyelids
    )
    #model.jawTrans0 = jawTrans0

    output_list = dataset.output_list
    for params in tqdm(output_list):

        jawTrans = flame.get_jaw_transfer(
            cameras=torch.inverse(params.cameras.R.to(args.data_device)),
            shape_params=params.shape,
            expression_params=params.exp,
            eye_pose_params=params.eyes,  # 12
            jaw_pose_params=params.jaw,  # 6
            eyelid_params=params.eyelids  # 2
        ) # 4x4
        # p
        rjawTrans = torch.eye(4,device=jawTrans.device) + jawTrans - jawTrans0
        rjawTrans = rjawTrans.unsqueeze(0) # 1x4x4
        rjawTransN = normalize_transform(rjawTrans[:,:3,:3]) # 1x3x3

        params.rjawTrans = rjawTrans
        params.rjawTransN = rjawTransN


@torch.no_grad()
def rigid_transfer(dataset, model, args, gen_local_frame = True, gpu_side = True):
    from .rigid_transform_3D import rigid_transform_3D

    flame = model.flame
    faces = model.faces

    back_head_face_idx = np.loadtxt(args.back_head_file)
    back_head_face_idx = np.asarray(back_head_face_idx, dtype=np.int32)
    back_head_vertex_idx = get_all_touched_index(faces, back_head_face_idx)

    dummy_frame = dataset.output_list[0]

    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    vertices0, _, _ = flame(
        cameras = torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params = dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,
        jaw_pose_params=zeros_jaw,
        eyelid_params=zeros_eyelids
    )
    vertices0 = vertices0[0] # BxPx3 -> Px3
    points_neutral = vertices0[back_head_vertex_idx]
    points_neutral = points_neutral.cpu().detach().numpy()
    if gen_local_frame:
        f_id_xyz0 = vertices0[faces].to(args.data_device) # Fx3(id)x3(xyz) F=5k, triangle faces
        local_frame = create_local_frame(f_id_xyz0).to(args.data_device)
        model.vertices0 = vertices0
        model.local_frame = local_frame

    output_list = dataset.output_list
    for params in tqdm(output_list):

        vertices, _, _ = flame(
            cameras=torch.inverse(params.cameras.R.to(args.data_device)),
            shape_params=params.shape,
            expression_params=params.exp,
            eye_pose_params=params.eyes,  # 12
            jaw_pose_params=params.jaw,  # 6
            eyelid_params=params.eyelids  # 2
        )
        vertices = vertices[0] # BxPx3 -> Px3
        #vertices_frame = vertices.cpu().detach().numpy()
        points_frame = vertices[back_head_vertex_idx]
        points_frame = points_frame.cpu().detach().numpy()
        R, t = rigid_transform_3D(points_neutral.T, points_frame.T) # neutral->exp
        #print('R t',R.shape,t.shape)
        R = torch.from_numpy(R).float()
        t = torch.from_numpy(t).float()
        if gpu_side:
            R = R.cuda()
            t = t.cuda()

        params.frameR = R
        params.framet = t
        params.frameNR = R


@torch.no_grad()
def from_mesh_to_point(dataset, model, args, merge_iden=True):

    flame = model.flame
    faces = model.faces
    face_id = model.face_id # Px3
    barycentric_coord = model.barycentric_coord # Px3

    p_face_id = faces[face_id] # Px3
    output_list = dataset.output_list

    dummy_frame = dataset.output_list[0]

    ### pre-compute blendshape
    # v_template
    v_template = flame.v_template # Nx3
    v_template = v_template[p_face_id] # {Px3}x3
    p_template = (v_template * barycentric_coord[...,None]).sum(-2)

    # shape blendshape
    shape_v_tensor = flame.shapedirs # Nx3x400
    shape_p_tensor = shape_v_tensor[p_face_id] # {Px3}x3x400
    shape_p_tensor = (shape_p_tensor * barycentric_coord[...,None,None]).sum(1)
    if merge_iden:
        iden = dummy_frame.shape
        offset_ = (shape_p_tensor[...,:300] * iden).sum(-1)
        p_template = p_template + offset_
        shape_p_tensor = shape_p_tensor[...,300:]

    # pose blendshape
    pose_v_tensor = flame.posedirs # 36x(Vx3)
    pose_v_tensor = pose_v_tensor.view(pose_v_tensor.shape[0],-1,3) # 36xVx3
    pose_v_tensor = pose_v_tensor[:,p_face_id] # 36x{Px3}x3
    pose_p_tensor = (pose_v_tensor * barycentric_coord[...,None]).sum(-2) # 36xPx3
    pose_p_tensor = pose_p_tensor.view(pose_p_tensor.shape[0],-1) # 36x(Px3)

    # eyelid blendshape
    l_eyelid_v_tensor = flame.l_eyelid # 1x5023x3
    l_eyelid_v_tensor = l_eyelid_v_tensor[0][p_face_id] # {5023x3} x 3
    l_eyelid_p_tensor = (l_eyelid_v_tensor * barycentric_coord[...,None]).sum(-2)

    r_eyelid_v_tensor = flame.r_eyelid  # 1x5023x3
    r_eyelid_v_tensor = r_eyelid_v_tensor[0][p_face_id]  # {5023x3} x 3
    r_eyelid_p_tensor = (r_eyelid_v_tensor * barycentric_coord[..., None]).sum(-2)

    # lbs_weights
    lbs_weights_v_tensor = flame.lbs_weights # 5023x3
    lbs_weights_v_tensor = lbs_weights_v_tensor[p_face_id] # {5023x3} x 3
    lbs_weights_p_tensor = (lbs_weights_v_tensor * barycentric_coord[...,None]).sum(-2)

    model.v_template = p_template # Px3
    model.shapedirs = shape_p_tensor # Px3x{400/100}
    model.posedirs = pose_p_tensor # 36x(Px3)
    model.l_eyelid = l_eyelid_p_tensor # Px3
    model.r_eyelid = r_eyelid_p_tensor # Px3
    model.lbs_weights = lbs_weights_p_tensor # Px5

    zeros_exp = torch.zeros_like(dummy_frame.exp)
    zeros_eyes = torch.tensor([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_jaw = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(args.data_device)
    zeros_eyelids = torch.tensor([[0, 0]], dtype=torch.float32).to(args.data_device)

    dummy_frame = output_list[0]
    J_transformed0, A0 = flame.get_transfer(
        cameras=torch.inverse(dummy_frame.cameras.R.to(args.data_device)),
        shape_params=dummy_frame.shape,
        expression_params=zeros_exp,
        eye_pose_params=zeros_eyes,  # 12
        jaw_pose_params=zeros_jaw,  # 6
        eyelid_params=zeros_eyelids  # 2
    )
    for params in tqdm(output_list):
        J_transformed, A = flame.get_transfer(
            cameras=torch.inverse(params.cameras.R.to(args.data_device)),
            shape_params=params.shape,
            expression_params=params.exp,
            eye_pose_params=params.eyes,  # 12
            jaw_pose_params=params.jaw,  # 6
            eyelid_params=params.eyelids  # 2
        )
        trans = torch.eye(4,device=A.device) + A - A0
        #transN = normalize_transform(trans[:,:3,:3])
        # normalize
        # J_transformed = 1x5x3
        # A = 1x5x4x4
        params.J_transformed = J_transformed
        params.A = A
        params.A_diff = trans
