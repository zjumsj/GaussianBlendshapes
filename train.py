import os
import numpy as np
import uuid
from tqdm import tqdm
from argparse import Namespace
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.loss_utils import l1_loss_pixel, l1_loss, ssim
from utils.utils import str2bool, dump_code, images_to_video
from utils.general_utils import safe_state
from utils.image_utils import psnr

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

dump_profiler = False
#dump_profiler = True
if dump_profiler:
    import torch.profiler

def mask_function(x,args):
    threshold = max_displacement_of_blendshape49 * 0.1
    L = torch.sqrt(torch.clamp(torch.sum(x * x, dim=1),1e-18,None))
    y = torch.clamp((L-threshold) / (max_displacement_of_blendshape0 - threshold),0,None)
    return y


def config_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Train your network sailor.')
    parser.add_argument('--sh_degree',type=int,default=config.sh_degree,help='sh level total basis is (D+1)*(D+1)')
    parser.add_argument('-s','--source_path',type=str,default=config.source_path, help='dataset path')
    parser.add_argument('-m','--model_path',type=str, default=config.model_path, help='model path')
    parser.add_argument("--white_bkgd", type=str2bool, default=config.white_bkgd, help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--data_device",type=str,default=config.data_device)
    parser.add_argument("--reside_image_on_gpu",type=str2bool,default=config.reside_image_on_gpu)
    parser.add_argument("--use_nerfBS",type=str2bool, default=config.use_nerfBS, help='enable to train on NeRFBlendShape dataset')
    parser.add_argument("--use_HR", type=str2bool, default=config.use_HR, help='use high resolution images')

    # optimizer
    parser.add_argument("--iterations", type=int, default=config.iterations)
    parser.add_argument("--position_lr_init", type=float, default=config.position_lr_init)
    parser.add_argument("--position_lr_final", type=float, default=config.position_lr_final)
    parser.add_argument("--position_lr_delay_mult", type=float, default=config.position_lr_delay_mult)
    parser.add_argument("--position_lr_max_steps", type=int, default=config.position_lr_max_steps)

    parser.add_argument("--feature_lr",type=float, default=config.feature_lr)
    parser.add_argument("--opacity_lr",type=float, default=config.opacity_lr)
    parser.add_argument("--scaling_lr",type=float, default=config.scaling_lr)
    parser.add_argument("--rotation_lr",type=float, default=config.rotation_lr)
    parser.add_argument("--percent_dense",type=float, default=config.percent_dense)
    parser.add_argument("--lambda_dssim", type=float, default=config.lambda_dssim)

    parser.add_argument("--densification_interval", type=int, default=config.densification_interval)
    parser.add_argument("--opacity_reset_interval", type=int, default=config.opacity_reset_interval)
    parser.add_argument("--densify_from_iter", type=int, default=config.densify_from_iter)
    parser.add_argument("--densify_until_iter", type=int, default=config.densify_until_iter)
    parser.add_argument("--densify_grad_threshold", type=float, default=config.densify_grad_threshold)

    parser.add_argument("--camera_extent", type=float, default=config.camera_extent)
    parser.add_argument("--convert_SHs_python", type=str2bool, default=config.convert_SHs_python)
    parser.add_argument("--compute_cov3D_python",type=str2bool, default=config.compute_cov3D_python)
    parser.add_argument("--debug",type=str2bool,default=False)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=list, default=config.test_iterations)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=config.checkpoint_iterations)

    # face
    parser.add_argument('--flame_geom_path', type=str, default=config.flame_geom_path)
    parser.add_argument('--flame_lmk_path', type=str, default=config.flame_lmk_path)
    parser.add_argument('--back_head_file', type=str, default=config.back_head_file)

    parser.add_argument('--use_dyn_point', type=bool, default=config.use_dyn_point)
    parser.add_argument('--update_consistency', type=str2bool, default=config.update_consistency)
    parser.add_argument('--init_face_point_number', type=int, default=config.init_face_point_number)
    parser.add_argument('--num_shape_params', type=int, default=config.num_shape_params)
    parser.add_argument('--num_exp_params',type=int ,default=config.num_exp_params)

    parser.add_argument('--basis_lr_decay',type=float,default=config.basis_lr_decay)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)
    parser.add_argument('--alpha_loss',type=float, default=config.alpha_loss)
    parser.add_argument('--mouth_loss_weight', type=float, default=config.mouth_loss_weight)
    parser.add_argument('--mouth_loss_type', type=float, default=config.mouth_loss_type)
    parser.add_argument('--cylinder_params', type=object, default=config.cylinder_params)
    parser.add_argument('--isotropic_loss',type=float, default=config.isotropic_loss)
    parser.add_argument('--lpips_loss',type=float, default=config.lpips_loss)

    args, unknown = parser.parse_known_args()

    if len(unknown) != 0:
        print(unknown)
        exit(-1)

    args.flame_template_path = os.path.join(args.source_path,"canonical.obj")

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    os.makedirs(args.model_path,exist_ok=True)
    dump_code(os.path.dirname(os.path.abspath(__file__)), args.model_path)

    return args

def training(args, testing_iterations, checkpoint_iterations, debug_from):

    first_iter = 0
    if args.use_nerfBS:
        dataset = FaceDatasetNerfBS(args.source_path, shuffle=False)
        dataset.prepare_data(reside_image_on_gpu=args.reside_image_on_gpu,device=args.data_device)
    else:
        if args.use_HR:
            dataset = FaceDatasetDyn(args.source_path, shuffle=False, ratio=2.0)
            dataset.prepare_data(reside_image_on_gpu=args.reside_image_on_gpu,device=args.data_device)
            dataset.load_test_images_in_adv()
        else:
            dataset = FaceDataset(args.source_path, shuffle=False)
            dataset.prepare_data(reside_image_on_gpu=args.reside_image_on_gpu,device=args.data_device)
    dummy_frame = dataset.output_list[0]

    tb_writer = prepare_output_and_logger(args)

    face_gaussians = f_gaussian_model.GaussianModel(args.sh_degree)
    face_gaussians.create_from_face(dummy_frame, args, args.camera_extent)
    face_gaussians.training_setup(args, id=0)

    mouth_file0 = "./data/up_billboard_tri.obj"
    mouth_file1 = "./data/down_billboard_tri.obj"
    mouth_offsetfile = os.path.join(args.source_path,"offset.txt")
    if os.path.exists(mouth_offsetfile):
        mouth_offset = np.loadtxt(mouth_offsetfile,dtype=np.float32)
    else:
        mouth_offset = np.array([3.3226e-4, 2.29566e-3, -1.21933e-3],dtype=np.float32)

    cylinder_params = args.cylinder_params
    bounding_cylinder = mouth_model.BoundingCylinder(
        np.array(cylinder_params[:3],dtype=np.float32) + mouth_offset,
        R=cylinder_params[3], half_H=cylinder_params[4]
    )

    mouth_gaussians_up = mouth_model.GaussianModel(args.sh_degree,0) # up teeth
    mouth_gaussians_down = mouth_model.GaussianModel(args.sh_degree,1) # down teeth

    ##
    mouth_gaussians_up.create_from_face(mouth_file0, mouth_offset, args, args.camera_extent)
    mouth_gaussians_down.create_from_face(mouth_file1, mouth_offset, args, args.camera_extent)

    mouth_gaussians_up.training_setup(args)
    mouth_gaussians_down.training_setup(args)

    ## Generate rigid transfer according to anchor points on the back of the head
    # used to transfer upper teeth
    f_transforms.rigid_transfer(dataset, face_gaussians, args, gen_local_frame=False)
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

    print('initialize ...')
    #face_gaussians.extract_acc()
    face_gaussians.compute_blendshape_init()

    bg_color = [1, 1, 1] if args.white_bkgd else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    if dump_profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.model_path),
            record_shapes=True,
            with_stack=True
        )
        prof.start()

    lpips_loss_fn = None
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, args.iterations + 1):

        iter_start.record()

        face_gaussians.update_learning_rate(iteration)
        mouth_gaussians_up.update_learning_rate(iteration)
        mouth_gaussians_down.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            face_gaussians.oneupSHdegree()
            mouth_gaussians_up.oneupSHdegree()
            mouth_gaussians_down.oneupSHdegree()

        # Pick a random Camera
        if args.use_HR:
            if not viewpoint_stack:
                viewpoint_stack = dataset.getTrainCameras().copy()
                random.shuffle(viewpoint_stack)
                dataset.create_load_seqs(viewpoint_stack)
                viewpoint_stack = viewpoint_stack.copy()
            viewpoint_cam = viewpoint_stack.pop(0)
            frame = dataset.getData(viewpoint_cam, load_mode='load')
        else:
            if not viewpoint_stack:
                viewpoint_stack = dataset.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
            frame = dataset.getData(viewpoint_cam)

        # Render
        if (iteration - 1) == debug_from:
            args.debug = True
        face_gaussians.prepare_merge(frame)
        face_gaussians.prepare_xyz(frame,args)

        mouth_gaussians_up.prepare_xyz(frame,args)
        mouth_gaussians_down.prepare_xyz(frame,args)

        gaussians = [face_gaussians, mouth_gaussians_up, mouth_gaussians_down]

        render_pkg = f_renderer.render_alpha(frame, gaussians, args, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        alpha0 = render_pkg['alpha0']

        # Loss
        gt_image = frame.original_image.cuda()
        gt_image = gt_image.permute(2,0,1)

        if args.use_nerfBS:
            bkg = frame.bkg.cuda()
            bkg = bkg.permute(2,0,1)
            mask = frame.mask.cuda()
            image = image + (1-alpha0) * bkg # image with background
            gt_image_ = gt_image
        else:
            mask = frame.mask.cuda()
            gt_image_ = gt_image * mask
        Ll1 = l1_loss(image, gt_image_)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim(image, gt_image_))

        if args.lpips_loss:
            if lpips_loss_fn is None:
                import lpips
                #lpips_loss_fn =  lpips.LPIPS(net='alex')
                lpips_loss_fn = lpips.LPIPS(net='vgg')
                lpips_loss_fn = lpips_loss_fn.to(args.data_device)
            # image should be RGB and normalized to [-1,1]
            d = lpips_loss_fn(
                gt_image_.unsqueeze(0) * 2. - 1.,
                image.unsqueeze(0) * 2. - 1.
            )
            loss = loss + d * args.lpips_loss

        if args.isotropic_loss: # add isotropic constraints
            gaussians_scale = [o.get_scaling for o in gaussians]
            gaussians_scale = torch.cat(gaussians_scale,dim=0)
            max_scale = gaussians_scale.max(-1)[0]
            min_scale = gaussians_scale.min(-1)[0]
            ratio = (max_scale + 1e-3) / (min_scale + 1e-3)
            tmp = (ratio - 1.) ** 2.
            loss = loss + args.isotropic_loss * tmp.mean()

        if args.mouth_loss_type == 1:
            loss = loss + args.mouth_loss_weight * (
                bounding_cylinder.compute_loss(mouth_gaussians_up._xyz).mean() +
                bounding_cylinder.compute_loss(mouth_gaussians_down._xyz).mean()
            )
        elif args.mouth_loss_type == 2:
            bounding_cylinder.hard_constraint(mouth_gaussians_up._xyz)
            bounding_cylinder.hard_constraint(mouth_gaussians_down._xyz)

        if args.alpha_loss > 0:
            loss = loss + args.alpha_loss * ((alpha0 - mask) ** 2)


        loss = loss.mean()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == args.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss_pixel, iter_start.elapsed_time(iter_end), testing_iterations, gaussians, dataset,  (args, background))

            # Densification
            if args.use_dyn_point:
                if iteration < args.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    n_points = [
                        face_gaussians._scaling.shape[0],
                        mouth_gaussians_up._scaling.shape[0],
                        mouth_gaussians_down._scaling.shape[0],
                    ]
                    radiis = torch.split(radii,n_points,dim=0)
                    visibility_filters = torch.split(visibility_filter,n_points,dim=0)

                    grads = torch.split(viewspace_point_tensor.grad, n_points, dim=0)
                    viewspace_point_tensors = torch.split(viewspace_point_tensor, n_points, dim=0)
                    for g,v in zip(grads, viewspace_point_tensors):
                        v.grad = g

                    face_gaussians.max_radii2D[visibility_filters[0]] = torch.max(face_gaussians.max_radii2D[visibility_filters[0]], radiis[0][visibility_filters[0]])
                    face_gaussians.add_densification_stats(viewspace_point_tensors[0], visibility_filters[0])

                    mouth_gaussians_up.max_radii2D[visibility_filters[1]] = torch.max(mouth_gaussians_up.max_radii2D[visibility_filters[1]], radiis[1][visibility_filters[1]])
                    mouth_gaussians_up.add_densification_stats(viewspace_point_tensors[1], visibility_filters[1])

                    mouth_gaussians_down.max_radii2D[visibility_filters[2]] = torch.max(mouth_gaussians_down.max_radii2D[visibility_filters[2]], radiis[2][visibility_filters[2]])
                    mouth_gaussians_down.add_densification_stats(viewspace_point_tensors[2], visibility_filters[2])

                    if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                        size_threshold = 20 if iteration > args.opacity_reset_interval else None
                        #face_gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, args.camera_extent, size_threshold)
                        face_gaussians.densify_and_prune(args.densify_grad_threshold * 5, 0.005, args.camera_extent, size_threshold)
                        mouth_gaussians_up.densify_and_prune(args.densify_grad_threshold, 0.005, args.camera_extent, size_threshold)
                        mouth_gaussians_down.densify_and_prune(args.densify_grad_threshold, 0.005, args.camera_extent, size_threshold)

                    if iteration % args.opacity_reset_interval == 0 or (args.white_bkgd and iteration == args.densify_from_iter):
                        pass

            if args.update_consistency and (iteration % args.densification_interval == 0 and iteration < args.iterations):
                face_gaussians.update_from_nn(update_acc_dict=True)

            # Optimizer step
            if iteration < args.iterations:
                face_gaussians.optimizer.step()
                face_gaussians.optimizer.zero_grad(set_to_none = True)
                mouth_gaussians_up.optimizer.step()
                mouth_gaussians_up.optimizer.zero_grad(set_to_none = True)
                mouth_gaussians_down.optimizer.step()
                mouth_gaussians_down.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((face_gaussians.capture(), iteration), args.model_path + "/fix_chkpnt" + str(iteration) + ".pth")
                face_gaussians.save_acc_dict(args.model_path + "/acc" + str(iteration) + ".npy")
                torch.save((mouth_gaussians_up.capture(), iteration), args.model_path + "/mouth0_chkpnt" + str(iteration) + ".pth")
                torch.save((mouth_gaussians_down.capture(), iteration), args.model_path + "/mouth1_chkpnt" + str(iteration) + ".pth")

        if dump_profiler:
            prof.step()
            if iteration == 30:
                break

    if dump_profiler:
        prof.stop()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, gaussians, dataset, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        tmp1 = dataset.getTestCameras()
        if renderArgs[0].use_HR:
            tmp1 = [dataset.getData(o, load_mode='dont_load') for o in tmp1]
        else:
            tmp1 = [dataset.getData(o) for o in tmp1]
        tmp2 = dataset.getTrainCameras()
        tmp2_idx = [o % len(tmp2) for o in range(5,30,5)]
        tmp2 = [tmp2[o] for o in tmp2_idx]
        tmp2 = [dataset.getData(o) for o in tmp2]

        validation_configs = ({"name":"test","cameras":tmp1},{"name":"train", "cameras":tmp2})
        del tmp1, tmp2, tmp2_idx

        face_gaussians = gaussians[0]
        mouth_gaussians_up = gaussians[1]
        mouth_gaussians_down = gaussians[2]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    face_gaussians.prepare_merge(viewpoint)
                    face_gaussians.prepare_xyz(viewpoint,renderArgs[0])

                    mouth_gaussians_up.prepare_xyz(viewpoint,renderArgs[0])
                    mouth_gaussians_down.prepare_xyz(viewpoint,renderArgs[0])

                    image_set = f_renderer.render_alpha(viewpoint, gaussians, *renderArgs)

                    pred_mask0 = image_set["alpha0"]
                    gt_image = viewpoint.original_image.to("cuda")
                    gt_image = gt_image.permute(2, 0, 1)
                    if renderArgs[0].use_nerfBS:
                        bkg = viewpoint.bkg.cuda()
                        bkg = bkg.permute(2, 0, 1)
                        image = image_set['render']
                        image = image + (1 - pred_mask0) * bkg
                        image = torch.clamp(image, 0.0, 1.0)
                    else:
                        mask = viewpoint.mask.to("cuda")
                        gt_image = torch.clamp(gt_image * mask, 0.0, 1.0)
                        image = torch.clamp(image_set["render"], 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        ## draw extra
                        views = face_gaussians.vis(viewpoint,renderArgs[0],[f_gaussian_model.View.SHAPE])
                        tb_writer.add_image(config['name'] + "_view_{}/mesh".format(viewpoint.image_name), np.clip(views[0],0,1), global_step=iteration, dataformats='HWC')
                        ### draw mask
                        face_mask = torch.stack([viewpoint.mask, torch.zeros_like(viewpoint.mask), torch.zeros_like(viewpoint.mask)], dim=0)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/gt_mask".format(viewpoint.image_name), torch.clip(face_mask,0,1), global_step=iteration, dataformats='CHW')
                        pred_face_mask = torch.stack([pred_mask0, torch.zeros_like(pred_mask0), torch.zeros_like(pred_mask0)],dim=0)
                        tb_writer.add_images(config['name'] + "_view_{}/pred_mask".format(viewpoint.image_name), torch.clip(pred_face_mask,0,1), global_step=iteration, dataformats='CHW')
                        ### draw mouth
                        mouth_image = f_renderer.render(viewpoint, [mouth_gaussians_up, mouth_gaussians_down], *renderArgs)["render"]
                        mouth_image = torch.clamp(mouth_image,0.0,1.0)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth".format(viewpoint.image_name), mouth_image, global_step=iteration, dataformats='CHW')

                        pc_numbers = [
                            mouth_gaussians_up._scaling.shape[0],
                            mouth_gaussians_down._scaling.shape[0],
                        ]
                        override_color = [
                            torch.tensor([1.,0.,0.],device="cuda").repeat(pc_numbers[0],1),
                            torch.tensor([0.,1.,0.],device="cuda").repeat(pc_numbers[1],1),
                        ]
                        mouth_mask =  f_renderer.render(viewpoint, [mouth_gaussians_up, mouth_gaussians_down], *renderArgs, override_color=override_color)["render"]
                        mouth_mask = torch.clamp(mouth_mask, 0.0, 1.0)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name),mouth_mask, global_step=iteration, dataformats='CHW')

                    ## 
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            for ii in range(len(gaussians)):
                tb_writer.add_histogram("scene/opacity_histogram_%d" % ii, gaussians[ii].get_opacity, iteration)
                tb_writer.add_scalar('total_points_%d' % ii, gaussians[ii].get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def train():

    args = config_parse()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    os.makedirs(args.model_path,exist_ok=True)
    dump_code(os.path.dirname(os.path.abspath(__file__)), args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, args.test_iterations, args.checkpoint_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    train()

