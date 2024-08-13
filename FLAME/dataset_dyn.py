'''
    Differences between dataset.py & dataset_dyn.py
    This loader loads images on-the-fly instead of loading once at initialization
    The strategy saves memory, especially when using high-resolution images (e.g. 1024x)
    Otherwise, it may consume ~100G memory.
'''
import os
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import rotation_6d_to_matrix

import copy

import numpy as np
import torch
import glob
import cv2

num_workers = 8

class DummyObj:
    def __init__(self):
        pass

def collate_fn(data):
    output = []
    n = len(data)
    n_item = len(data[0])
    for ii in range(n_item):
        o = [data[jj][ii] for jj in range(n)]
        if type(o[0]) == type(0):
            o = o[0]
        elif o[0] is not None:
            o = torch.stack(o, dim=0)
        else:
            o = None
        output.append(o)

    return output

def image_loader(path, id, ratio):
    if ratio > 1.:
        image = cv2.imread(os.path.join(path, "images.HQ/%05d.png" % id), cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(os.path.join(path, "images/%05d.png" % id), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = np.asarray(image, dtype=np.float32) / 255.
    image_mask = image[..., -1]
    image = image[..., :3]

    image = torch.from_numpy(image)
    image_mask = torch.from_numpy(image_mask)

    return image, image_mask

class ImageDataLoader(Dataset):
    def __init__(self, path, idx_list, ratio):
        self.path = path
        self.idx_list = idx_list
        self.ratio = ratio

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        image, image_mask = image_loader(self.path, self.idx_list[idx], self.ratio)
        return image, image_mask

class FaceDatasetDyn:
    def __init__(self, dataset_name, load_iterations=None, shuffle=True, ratio=2.0):

        # We always run face tracker in downsampled images (i.e. 512x)
        # However the images for training may be in higher resolution, such as 1024x
        # Some special cares need to be taken
        # for 512x images, set ratio=1.0
        # for 1024x images, set ratio=2.0
        # and so on
        self.ratio = ratio

        self.dataset_name = dataset_name
        file_list = glob.glob(os.path.join(dataset_name, "checkpoint", "*.frame"))
        self.shuffle = shuffle
        self.n_frames = len(file_list)
        self.n_seg = 350 # use last 350 frames as test set
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

        ### try to load a frame to determin H,W
        if self.ratio > 1.:
            image = cv2.imread(os.path.join(self.dataset_name, "images.HQ/%05d.png" % 0))
        else:
            image = cv2.imread(os.path.join(self.dataset_name, "images/%05d.png" % 0))
        self.H, self.W, _ = image.shape

    def getTrainCameras(self):
        return self.train_ids

    def getTestCameras(self):
        return self.test_ids

    def prepare_data(self, reside_image_on_gpu=True, device="cuda"):
        output_list = []
        for ii in range(self.n_frames):
            output_list.append(self.getData(ii,reside_image_on_gpu,device))
        self.output_list = output_list

    def create_load_seqs(self, idx_list):
        # create new loader here
        dataset = ImageDataLoader(self.dataset_name, idx_list, self.ratio)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, prefetch_factor=num_workers
        )
        self.dataloader_iterable = iter(dataloader)

    def load_test_images_in_adv(self):
        for id in self.test_ids:
            obj = self.output_list[id]
            image, image_mask = image_loader(self.dataset_name, id, self.ratio)
            obj.original_image = image
            obj.mask = image_mask

    def getData(self, id, reside_image_on_gpu=True, device="cuda", load_mode='load_directly'):

        if self.output_list is not None:
            obj = self.output_list[id]
            ### make a shallow copy and append images
            obj_copy = copy.copy(obj)
            if load_mode == 'load_directly':
                image, image_mask = image_loader(self.dataset_name, id, self.ratio)
                obj_copy.original_image = image
                obj_copy.mask = image_mask
            elif load_mode == 'load':
                image_data = next(self.dataloader_iterable)
                # remove batch dim
                image_data = [(o[0] if o is not None else None) for o in image_data]
                image, image_mask = image_data
                obj_copy.original_image = image
                obj_copy.mask = image_mask
            elif load_mode == 'dont_load':
                pass
            else:
                raise NotImplementedError

            return obj_copy

        H, W = self.H, self.W

        frame = torch.load(os.path.join(self.dataset_name,"checkpoint/%05d.frame" % id))
        frame_flame = frame['flame']
        frame_camera = frame['camera']
        frame_opencv = frame['opencv']

        camera = PerspectiveCameras(
            device=device,
            principal_point=torch.from_numpy(frame_camera['pp']).to(device),
            focal_length=torch.from_numpy(frame_camera['fl']).to(device),
            R=rotation_6d_to_matrix(torch.from_numpy(frame_camera['R']).to(device)),
            T=torch.from_numpy(frame_camera['t']).to(device),
            image_size=[[H, W]]
        )
        output = DummyObj()
        # BASE
        output.original_image = None
        output.mask = None
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

        w2c = np.zeros([4, 4])
        w2c[3, 3] = 1
        w2c[:3, :3] = frame_opencv['R'][0]
        w2c[:3, 3] = frame_opencv['t'][0]

        c2w = np.linalg.inv(w2c)

        t_w2c = torch.from_numpy(w2c.transpose()).float().to(device)
        t_c2w = torch.from_numpy(c2w.transpose()).float().to(device)

        znear = 0.01
        zfar = 100.0
        z_sign = 1.0
        proj = np.zeros([4, 4])
        w,h = W/self.ratio, H/self.ratio
        proj[0, :2] = frame_opencv['K'][0, 0, :2] * 2. / w
        proj[1, :2] = frame_opencv['K'][0, 1, :2] * 2. / h
        proj[0, 2] = frame_opencv['K'][0, 0, 2] * 2. / w - 1.
        proj[1, 2] = frame_opencv['K'][0, 1, 2] * 2. / h - 1.
        proj[3, 2] = z_sign
        proj[2, 2] = z_sign * zfar / (zfar - znear)
        proj[2, 3] = -(zfar * znear) / (zfar - znear)

        proj_w2c = proj @ w2c
        t_proj_w2c = torch.from_numpy(proj_w2c.transpose()).float().to(device)

        output.FoVx = 2 * np.arctan(w * 0.5 / frame_opencv['K'][0, 0, 0])
        output.FoVy = 2 * np.arctan(h * 0.5 / frame_opencv['K'][0, 1, 1])
        output.image_height = H  # real image size
        output.image_width = W
        output.world_view_transform = t_w2c.contiguous()
        output.full_proj_transform = t_proj_w2c.contiguous()
        output.camera_center = t_c2w[3, :3].contiguous()

        return output


