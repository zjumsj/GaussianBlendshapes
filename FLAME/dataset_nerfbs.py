import os
import random

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import rotation_6d_to_matrix

import numpy as np
import torch
import glob
import cv2


class DummyObj:
    def __init__(self):
        pass

class FaceDatasetNerfBS:
    def __init__(self, dataset_name, load_iterations = None, shuffle=True, resolution_scale=[1.0]):
        self.dataset_name = dataset_name
        file_list = glob.glob(os.path.join(dataset_name,"checkpoint","*.frame"))
        self.shuffle = shuffle
        self.n_frames = len(file_list)
        self.n_seg = 500 # use last 500 frames as test set
        self.n_extract_ratio = -1
        train_ids = []
        test_ids = []
        for ii in range(self.n_frames):
            if ii + self.n_seg >= self.n_frames:
                test_ids.append(ii)
            else:
                train_ids.append(ii)
        self.train_ids = train_ids
        self.test_ids = test_ids
        if self.shuffle:
            random.shuffle(self.train_ids)
            random.shuffle(self.test_ids)
        self.output_list = None

        # background image
        img = cv2.imread(os.path.join(dataset_name,"bc.jpg"))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.asarray(img,dtype=np.float32)/255.
        img = torch.from_numpy(img).to("cuda")
        self.bkg = img

    def getTrainCameras(self):
        return self.train_ids

    def getTestCameras(self):
        return self.test_ids

    def prepare_data(self, reside_image_on_gpu=True, device="cuda"):
        output_list = []
        for ii in range(self.n_frames):
            output_list.append(self.getData(ii,reside_image_on_gpu,device))
        self.output_list = output_list

    def getData(self, id, reside_image_on_gpu=True ,device="cuda"):

        if self.output_list is not None:
            return self.output_list[id]

        ## image & image mask
        image = cv2.imread(os.path.join(self.dataset_name, "images/%05d.png" % id), cv2.IMREAD_UNCHANGED)
        image_mask = cv2.imread(os.path.join(self.dataset_name, "image_mask/mask_%05d.png" % id), cv2.IMREAD_UNCHANGED)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(image, dtype=np.float32) / 255.  # RGB,HxWx3
        image_mask = np.asarray(image_mask, dtype=np.float32) / 255.
        if image_mask.shape[-1] == 3:
            image_mask = image_mask[..., 0]  # HxW

        image = torch.from_numpy(image)
        image_mask = torch.from_numpy(image_mask)
        if reside_image_on_gpu:
            image = image.to(device)
            image_mask = image_mask.to(device)

        H,W, _ = image.shape

        frame = torch.load(os.path.join(self.dataset_name,"checkpoint/%05d.frame" % id))
        frame_flame = frame['flame']
        frame_camera = frame['camera']
        frame_opencv = frame['opencv']

        camera = PerspectiveCameras(
            device = device,
            principal_point= torch.from_numpy(frame_camera['pp']).to(device),
            focal_length= torch.from_numpy(frame_camera['fl']).to(device),
            R = rotation_6d_to_matrix(torch.from_numpy(frame_camera['R']).to(device)),
            T = torch.from_numpy(frame_camera['t']).to(device),
            image_size = [[H,W]]
        )
        output = DummyObj()
        # BASE
        output.original_image = image
        output.mask = image_mask
        output.image_name = id
        # FLAME params
        output.cameras = camera
        output.image_size = [H,W]
        output.shape = torch.from_numpy(frame_flame['shape']).to(device)
        output.exp = torch.from_numpy(frame_flame['exp']).to(device)
        output.tex = torch.from_numpy(frame_flame['tex']).to(device)
        output.eyes = torch.from_numpy(frame_flame['eyes']).to(device)
        output.jaw = torch.from_numpy(frame_flame['jaw']).to(device)
        output.eyelids = torch.from_numpy(frame_flame['eyelids']).to(device)

        output.R = rotation_6d_to_matrix(torch.from_numpy(frame_camera['R']).to(device))
        output.t = torch.from_numpy(frame_camera['t']).to(device)

        w2c = np.zeros([4,4])
        w2c[3,3] = 1
        w2c[:3,:3] = frame_opencv['R'][0]
        w2c[:3,3] = frame_opencv['t'][0]

        c2w = np.linalg.inv(w2c)

        t_w2c = torch.from_numpy(w2c.transpose()).float().to(device)
        t_c2w = torch.from_numpy(c2w.transpose()).float().to(device)

        znear = 0.01
        zfar = 100.0
        z_sign = 1.0
        proj = np.zeros([4,4])
        proj[0,:2] = frame_opencv['K'][0,0,:2] * 2. / W
        proj[1,:2] = frame_opencv['K'][0,1,:2] * 2. / H
        proj[0,2] = frame_opencv['K'][0,0,2] * 2. / W - 1.
        proj[1,2] = frame_opencv['K'][0,1,2] * 2. / H - 1.
        proj[3,2] = z_sign
        proj[2,2] = z_sign * zfar / (zfar - znear)
        proj[2,3] = -(zfar * znear) / (zfar - znear)

        proj_w2c = proj @ w2c
        t_proj_w2c = torch.from_numpy(proj_w2c.transpose()).float().to(device)

        output.FoVx = 2 * np.arctan(W * 0.5 / frame_opencv['K'][0,0,0])
        output.FoVy = 2 * np.arctan(H * 0.5 / frame_opencv['K'][0,1,1])
        output.image_height = H
        output.image_width = W
        output.world_view_transform = t_w2c.contiguous()
        output.full_proj_transform = t_proj_w2c.contiguous()
        output.camera_center = t_c2w[3,:3].contiguous()

        # background image
        output.bkg = self.bkg

        return output

