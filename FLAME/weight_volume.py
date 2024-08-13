import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_knn._C import getNearestFeature, getNearestTriangleID, getIdAndWeight

class WeightVolumeIndex(torch.nn.Module):
    def __init__(self, filename, vertex, faces, N=128, findNumber=16, padding = 1e-3):
        super(WeightVolumeIndex,self).__init__()
        if filename is not None:
            self.load(filename)
            return
        # vertex = Px3
        # faces = Px3
        if isinstance(N,int):
            Nx = N
            Ny = N
            Nz = N
        else:
            Nx,Ny,Nz = N
        bbox_min = torch.min(vertex,dim=0)[0] - padding
        bbox_max = torch.max(vertex,dim=0)[0] + padding
        lin_x = torch.linspace(bbox_min[0], bbox_max[0], Nx)
        lin_y = torch.linspace(bbox_min[1], bbox_max[1], Ny)
        lin_z = torch.linspace(bbox_min[2], bbox_max[2], Nz)
        Z,Y,X = torch.meshgrid(lin_z, lin_y, lin_x)
        coord = torch.stack([X,Y,Z],dim=-1) # DHWx3
        query_point = coord.view(-1,3)

        faces = faces.int()

        print('Generate grid. Please wait, it may take minutes ...')
        grid_id = getNearestTriangleID(
            vertex.cpu().contiguous(),
            faces.cpu().contiguous(),
            query_point.cpu().contiguous(),
            findNumber
        )
        grid_id = grid_id.cuda()
        grid_id = torch.reshape(grid_id,
                        [coord.shape[0],coord.shape[1],coord.shape[2]])
        # DxHxW
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.grid_size = [Nx,Ny,Nz]
        self.grid_id = grid_id.contiguous()

    def load(self, filename):
        obj_ = torch.load(filename)
        self.bbox_min = obj_["bbox_min"]
        self.bbox_max = obj_["bbox_max"]
        self.grid_size = obj_["grid_size"]
        self.grid_id = obj_["grid_id"].contiguous().cuda()

    def dump(self, filename):
        obj_ = {
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "grid_size": self.grid_size,
            "grid_id": self.grid_id
        }
        torch.save(obj_,filename)

    def forward(self, points):
        # points Px3
        # get id Px8, weight Px8
        assert(len(points.shape) == 2)
        points = (points - self.bbox_min)/ (self.bbox_max - self.bbox_min)
        points = points * 2. - 1. # [-1,1] in pytorch
        return getIdAndWeight(
            points.contiguous(),
            self.grid_id, True
        )

class WeightVolume(torch.nn.Module):
    def __init__(self, filename, vertex, features, faces, N=128, findNumber=16, padding = 1e-3):
        super(WeightVolume,self).__init__()
        if filename is not None:
            self.load(filename)
            return
        # vertex = Px3
        # faces = Px3
        if isinstance(N,int):
            Nx = N
            Ny = N
            Nz = N
        else:
            Nx,Ny,Nz = N
        bbox_min = torch.min(vertex,dim=0)[0] - padding
        bbox_max = torch.max(vertex,dim=0)[0] + padding
        lin_x = torch.linspace(bbox_min[0], bbox_max[0], Nx)
        lin_y = torch.linspace(bbox_min[1], bbox_max[1], Ny)
        lin_z = torch.linspace(bbox_min[2], bbox_max[2], Nz)
        Z,Y,X = torch.meshgrid(lin_z, lin_y, lin_x)
        coord = torch.stack([X,Y,Z],dim=-1) # DHWx3
        query_point = coord.view(-1,3)

        faces = faces.int()

        print('Generate grid. Please wait, it may take minutes ...')
        tar_feature = getNearestFeature(
            vertex.cpu().contiguous(),
            features.cpu().contiguous(),
            faces.cpu().contiguous(),
            query_point.cpu().contiguous(),
            findNumber
        )
        tar_feature = tar_feature.cuda()
        tar_feature = tar_feature.permute(1,0)
        tar_feature = torch.reshape(tar_feature,
            [-1,coord.shape[0],coord.shape[1], coord.shape[2]])
        # CxDHW
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.grid_size = [Nx,Ny,Nz]
        self.tar_feature = tar_feature

    def load(self, filename):
        obj_ = torch.load(filename)
        self.bbox_min = obj_["bbox_min"]
        self.bbox_max = obj_["bbox_max"]
        self.grid_size = obj_["grid_size"]
        self.tar_feature = obj_["tar_feature"].cuda()

    def dump(self, filename):
        obj_ = {
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "grid_size": self.grid_size,
            "tar_feature": self.tar_feature
        }
        torch.save(obj_,filename)

    def forward(self, points):
        assert(len(points.shape) == 2)
        n_points = points.shape[0]
        C = self.tar_feature.shape[0]
        points = (points - self.bbox_min)/ (self.bbox_max - self.bbox_min)
        points = points * 2. - 1. # [-1,1] in pytorch
        # points = B x 3
        out_features = F.grid_sample(
            self.tar_feature.unsqueeze(0),
            points[None,None,None,...],
            mode='bilinear', padding_mode='border',
            align_corners= True
        )
        out_features = out_features.view(C,n_points)
        out_features = out_features.permute(1,0)
        return out_features
