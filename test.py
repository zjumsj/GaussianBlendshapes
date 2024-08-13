import os
import numpy as np
from tqdm import tqdm
# import uuid
# from argparse import Namespace
# from random import randint

import torch
#from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import safe_state
from utils.utils import str2bool, dump_code, images_to_video

import config.config_blendshapes as config

import FLAME.transforms as f_transforms
import FLAME.face_gs_model as f_gaussian_model
import FLAME.mouth_gs_model as mouth_model
from FLAME.dataset import FaceDataset
from FLAME.dataset_dyn import FaceDatasetDyn
from FLAME.dataset_nerfbs import FaceDatasetNerfBS
import FLAME.face_renderer as f_renderer

#ignore_neck = False
ignore_neck = True
max_displacement_of_blendshape0 = 0.005703532602638
max_displacement_of_blendshape49 = 0.000237277025008

torch.set_num_threads(1)

def mask_function(x,args):
    threshold = max_displacement_of_blendshape49 * 0.1
    L = torch.sqrt(torch.clamp(torch.sum(x * x, dim=1),1e-18,None))
    y = torch.clamp((L-threshold) / (max_displacement_of_blendshape0 - threshold),0,None)
    return y

def config_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Train your network sailor.')
    parser.add_argument('--sh_degree', type=int, default=config.sh_degree, help='sh level total basis is (D+1)*(D+1)')
    parser.add_argument('-s', '--source_path', type=str, default=config.source_path, help='dataset path')
    parser.add_argument('-m', '--model_path', type=str, default=config.model_path, help='model path')
    parser.add_argument("--white_bkgd", type=str2bool, default=config.white_bkgd, help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--data_device", type=str, default=config.data_device)
    parser.add_argument("--reside_image_on_gpu", type=str2bool, default=config.reside_image_on_gpu)
    parser.add_argument("--use_nerfBS", type=str2bool, default=config.use_nerfBS, help='enable to train on NeRFBlendShape dataset')
    parser.add_argument("--use_HR", type=str2bool, default=config.use_HR, help='use high resolution images')

    # optimizer
    parser.add_argument("--iterations", type=int, default=config.iterations)
    parser.add_argument("--position_lr_init", type=float, default=config.position_lr_init)
    parser.add_argument("--position_lr_final", type=float, default=config.position_lr_final)
    parser.add_argument("--position_lr_delay_mult", type=float, default=config.position_lr_delay_mult)
    parser.add_argument("--position_lr_max_steps", type=int, default=config.position_lr_max_steps)

    parser.add_argument("--feature_lr", type=float, default=config.feature_lr)
    parser.add_argument("--opacity_lr", type=float, default=config.opacity_lr)
    parser.add_argument("--scaling_lr", type=float, default=config.scaling_lr)
    parser.add_argument("--rotation_lr", type=float, default=config.rotation_lr)
    parser.add_argument("--percent_dense", type=float, default=config.percent_dense)
    # parser.add_argument("--lambda_dssim", type=float, default=config.lambda_dssim)

    parser.add_argument("--camera_extent", type=float, default=config.camera_extent)
    parser.add_argument("--convert_SHs_python", type=str2bool, default=config.convert_SHs_python)
    parser.add_argument("--compute_cov3D_python", type=str2bool, default=config.compute_cov3D_python)
    parser.add_argument("--debug", type=str2bool, default=False)

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=list, default=config.test_iterations)
    parser.add_argument("--quiet", action="store_true")
    #parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=config.checkpoint_iterations)

    # face
    parser.add_argument('--flame_geom_path', type=str, default=config.flame_geom_path)
    parser.add_argument('--flame_lmk_path', type=str, default=config.flame_lmk_path)
    parser.add_argument('--back_head_file', type=str, default=config.back_head_file)

    parser.add_argument('--init_face_point_number', type=int, default=config.init_face_point_number)
    parser.add_argument('--num_shape_params', type=int, default=config.num_shape_params)
    parser.add_argument('--num_exp_params',type=int ,default=config.num_exp_params)

    parser.add_argument('--basis_lr_decay', type=float, default=config.basis_lr_decay)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)

    # test
    parser.add_argument('--dump_for_viewer', type=str2bool, default=True)
    parser.add_argument('--render_seq', type=str2bool, default=False)
    parser.add_argument('--render_train', type=str2bool, default=False)
    parser.add_argument('--render_test', type=str2bool, default=True)
    parser.add_argument('--put_text', type=str2bool, default=False)
    parser.add_argument('--load_iteration', type=int, default=-1) # Default = search newest

    args, unknown = parser.parse_known_args()

    if len(unknown) != 0:
        print(unknown)
        exit(-1)

    args.flame_template_path = os.path.join(args.source_path, "canonical.obj")

    print("Test " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    os.makedirs(args.model_path,exist_ok=True)
    #dump_code(os.path.dirname(os.path.abspath(__file__)), args.model_path)

    return args

def to_image(x):
    if isinstance(x,torch.Tensor):
        x = x.cpu().detach().numpy()
    x = np.clip(np.round(x * 255.),0.,255.)
    return np.asarray(x,dtype=np.uint8)

def render_set(model_path, name, iteration, views, gaussians, args, background):
    from os import makedirs
    import cv2

    if args.use_HR:
        dataset, views = views
        dataset.create_load_seqs(views)

    render_path = os.path.join(model_path, name, "split_{}".format(iteration))
    merge_path = os.path.join(model_path, name, "join_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    makedirs(merge_path, exist_ok=True)

    face_gaussians = gaussians[0]
    mouth_gaussians_up = gaussians[1]
    mouth_gaussians_down = gaussians[2]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if args.use_HR:
            view = dataset.getData(view, load_mode = 'load')
        face_gaussians.prepare_merge(view)
        face_gaussians.prepare_xyz(view, args)

        mouth_gaussians_up.prepare_xyz(view, args)
        mouth_gaussians_down.prepare_xyz(view, args)

        rendering = f_renderer.render_alpha(view, gaussians, args, background)
        if args.use_nerfBS:
            gt = view.original_image[:, :, 0:3]
            gt = to_image(gt)
            bkg = view.bkg.cuda()
            bkg = bkg.permute(2,0,1)
            image = rendering['render']
            alpha0 = rendering['alpha0']
            image = image + (1-alpha0) * bkg # image with background
            image = image.permute(1,2,0)
            image = to_image(image)
        else:
            gt = view.original_image[:, :, 0:3] * view.mask[:, :, None]
            gt = to_image(gt)
            image = rendering['render'].permute(1,2,0)
            image = to_image(image)
        vis = face_gaussians.vis(view, args, [f_gaussian_model.View.SHAPE])
        shape = to_image(vis[0])

        gt_mask = torch.stack([view.mask, torch.zeros_like(view.mask), torch.zeros_like(view.mask)], dim=-1)
        gt_mask = to_image(gt_mask)

        alpha0 = rendering['alpha0']
        alpha_image = torch.stack([alpha0, torch.zeros_like(alpha0), torch.zeros_like(alpha0)],dim=-1)
        alpha_image = to_image(alpha_image)

        cv2.imwrite(os.path.join(render_path,"gt_%05d.png" % idx),gt[...,::-1])
        cv2.imwrite(os.path.join(render_path,"pred_%05d.png" % idx),image[...,::-1])
        cv2.imwrite(os.path.join(render_path,"mesh_%05d.png" % idx), shape[...,::-1])

        cv2.imwrite(os.path.join(render_path,"gt_mask_%05d.png" % idx), gt_mask[...,::-1])
        cv2.imwrite(os.path.join(render_path,"pred_mask_%05d.png" % idx), alpha_image[...,::-1])

        mouth_image = f_renderer.render(view, [mouth_gaussians_up, mouth_gaussians_down], args, background)["render"]
        mouth_image = to_image(mouth_image.permute(1,2,0))
        pc_numbers = [
            mouth_gaussians_up._scaling.shape[0],
            mouth_gaussians_down._scaling.shape[0],
        ]
        override_color = [
            torch.tensor([1., 0., 0.], device="cuda").repeat(pc_numbers[0], 1),
            torch.tensor([0., 1., 0.], device="cuda").repeat(pc_numbers[1], 1),
        ]
        mouth_mask = f_renderer.render(view, [mouth_gaussians_up, mouth_gaussians_down], args, background, override_color=override_color)["render"]
        mouth_mask = to_image(mouth_mask.permute(1,2,0))
        cv2.imwrite(os.path.join(render_path,"mouth_%05d.png" % idx),mouth_image[...,::-1])
        cv2.imwrite(os.path.join(render_path,"mouth_mask_%05d.png" % idx),mouth_mask[...,::-1])

        m = np.concatenate([gt,image,shape],axis=1)
        if args.render_seq and args.put_text:
            if args.n_extract_ratio == -1:
                tag = "Test" if (idx + args.n_seg >= len(views)) else "Train"
            else:
                tag = "Test" if ((idx // args.n_seg) % args.n_extract_ratio == args.n_extract_ratio - 1) else "Train"
            cv2.putText(m, tag, (m.shape[1] - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(merge_path, "%05d.png" % idx), m[...,::-1])

    images_to_video(merge_path)


def searchForMaxIteration2(folder):
    saved_iters = []
    for fname in os.listdir(folder):
        file, ext = os.path.splitext(fname)
        if ext == '.pth':
            start_id = fname.rfind('chkpnt') + 6
            saved_iters.append(int(file[start_id:]))
    print(saved_iters)
    return max(saved_iters)

def render_sets(args, render_train : bool, render_test : bool, render_seq:bool):
    with torch.no_grad():

        if args.use_nerfBS:
            dataset = FaceDatasetNerfBS(args.source_path, shuffle=False)
        else:
            if args.use_HR:
                dataset = FaceDatasetDyn(args.source_path, shuffle=False, ratio=2.0)
            else:
                dataset = FaceDataset(args.source_path, shuffle=False)
        dataset.prepare_data(reside_image_on_gpu=args.reside_image_on_gpu, device=args.data_device)
        dummy_frame = dataset.output_list[0]

        args.n_seg = dataset.n_seg
        args.n_extract_ratio = dataset.n_extract_ratio

        face_gaussians = f_gaussian_model.GaussianModel(args.sh_degree)
        face_gaussians.create_from_face(dummy_frame, args, args.camera_extent)

        mouth_file0 = "./data/up_billboard_tri.obj"
        mouth_file1 = "./data/down_billboard_tri.obj"
        mouth_offsetfile = os.path.join(args.source_path, "offset.txt")
        if os.path.exists(mouth_offsetfile):
            mouth_offset = np.loadtxt(mouth_offsetfile, dtype=np.float32)
        else:
            mouth_offset = np.array([3.3226e-4, 2.29566e-3, -1.21933e-3], dtype=np.float32)

        mouth_gaussians_up = mouth_model.GaussianModel(args.sh_degree, 0) # back of head only
        mouth_gaussians_down = mouth_model.GaussianModel(args.sh_degree, 1) # jaw

        mouth_gaussians_up.create_from_face(mouth_file0, mouth_offset, args, args.camera_extent)
        mouth_gaussians_down.create_from_face(mouth_file1, mouth_offset, args, args.camera_extent)

        ## Generate rigid transfer according to anchor points on the back of the head
        # used to transfer upper teeth
        f_transforms.rigid_transfer(dataset, face_gaussians, args, gen_local_frame=False)

        if args.load_iteration is None or args.load_iteration == -1:
            load_iteration = searchForMaxIteration2(args.model_path)
        else:
            load_iteration = args.load_iteration

        fix_checkpoint = os.path.join(args.model_path,"fix_chkpnt" + str(load_iteration) + ".pth")
        (model_params, first_iter) = torch.load(fix_checkpoint)
        face_gaussians.restore(model_params, args)

        fix_checkpoint2 = os.path.join(args.model_path,"acc" + str(load_iteration) + ".npy")
        face_gaussians.load_acc_dict(fix_checkpoint2)

        mouth_checkpoint0 = os.path.join(args.model_path, 'mouth0_chkpnt' + str(load_iteration) + ".pth")
        (model_params, first_iter) = torch.load(mouth_checkpoint0)
        mouth_gaussians_up.restore(model_params, args)
        #
        mouth_checkpoint1 = os.path.join(args.model_path, 'mouth1_chkpnt' + str(load_iteration) + ".pth")
        (model_params, first_iter) = torch.load(mouth_checkpoint1)
        mouth_gaussians_down.restore(model_params, args)

        ## Generate blendshape consistency scalar
        f_transforms.get_expr_consistency_face(face_gaussians, dummy_frame, mask_function, args, ignore_neck=ignore_neck)
        ## Generate deformation transfers for each expression blendshape
        f_transforms.get_expr_rot(face_gaussians, dummy_frame, args, light=True)
        ## Get pose blendshapes and eyelid blendshapes
        f_transforms.get_pose_tensor(face_gaussians, args)
        ## Get joints and joint transfers for each frame.
        f_transforms.from_mesh_to_point(dataset, face_gaussians, args)
        ## Generate jaw transfer
        # used to transfer lower teeth
        f_transforms.rigid_transfer_for_mouth2(dataset, face_gaussians, args)

        if args.dump_for_viewer: # Dump files for C++/CUDA viewer
            fix_checkpoint_path = os.path.splitext(fix_checkpoint)[0]
            face_gaussians.save_npy_forviewer(fix_checkpoint_path)
            print('dump npy %s' % fix_checkpoint_path)
            mouth_checkpoint_path0 = os.path.splitext(mouth_checkpoint0)[0]
            mouth_gaussians_up.save_npy_forviewer(mouth_checkpoint_path0)
            print('dump npy %s' % mouth_checkpoint_path0)
            mouth_checkpoint_path1 = os.path.splitext(mouth_checkpoint1)[0]
            mouth_gaussians_down.save_npy_forviewer(mouth_checkpoint_path1)
            print('dump npy %s' % mouth_checkpoint_path1)
            #return

        gaussians = [face_gaussians, mouth_gaussians_up, mouth_gaussians_down]

        bg_color = [1,1,1] if args.white_bkgd else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_ids = dataset.getTrainCameras()
        test_ids = dataset.getTestCameras()
        if args.use_HR:
            train_dataset = (dataset, train_ids)
            test_dataset = (dataset, test_ids)
            seq_dataset = (dataset, train_ids + test_ids)
        else:
            train_dataset = [dataset.output_list[i] for i in train_ids]
            test_dataset = [dataset.output_list[i] for i in test_ids]
            seq_dataset = dataset.output_list

        if render_seq:
            render_set(args.model_path, "seq", load_iteration, seq_dataset, gaussians, args, background)
        if render_train:
            render_set(args.model_path, "train", load_iteration, train_dataset, gaussians, args, background)
        if render_test:
            render_set(args.model_path, "test", load_iteration, test_dataset, gaussians, args, background)


def test():

    args = config_parse()
    safe_state(args.quiet)
    render_sets(args, args.render_train, args.render_test, args.render_seq)


if __name__ == "__main__":
    test()