
# model

sh_degree = 3
source_path = None
model_path = None

white_bkgd = False
data_device = "cuda:0"

reside_image_on_gpu = False

use_nerfBS = False
use_HR = False

# optimizer

iterations = 40_000
position_lr_init = 0.00016
position_lr_final = 0.0000016
position_lr_delay_mult = 0.01
position_lr_max_steps = 30_000

feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.005
rotation_lr = 0.001
percent_dense = 0.01
lambda_dssim = 0.2

densification_interval = 100
opacity_reset_interval = 3000
densify_from_iter = 500
densify_until_iter = 15_000
densify_grad_threshold = 0.0002

# Radius of the scene
camera_extent = 1.

# pipeline

convert_SHs_python = False
compute_cov3D_python = False
debug = False

test_iterations = [1,3000,10000,20000,30000,40000]
#save_iterations = [7_000, 30_000]
checkpoint_iterations = [40_000]

flame_geom_path = 'data/FLAME2020/generic_model.pkl'
flame_lmk_path = 'data/landmark_embedding.npy'
back_head_file = 'data/FLAME2020/back_of_head.txt'

use_dyn_point = True
# This flag corresponds to the dynamical updating of LBS blend weight and positional displacements
# described in Section 3.4 of the paper. However, this mechanism does not significantly affect the results.
# The quantitative results we reported in the paper were run with this flag disabled.
update_consistency = False
init_face_point_number = 50_000
num_shape_params = 300
num_exp_params = 100

basis_lr_decay = 1.
weight_decay = 0.

alpha_loss = 10

mouth_loss_weight = 100
# 1 for soft constraint, 2 for hard constraint
mouth_loss_type = 1
#mouth_loss_type = 2
cylinder_params = [0.0, -0.052537, 0.024792, 0.027279, 0.021632]

isotropic_loss = 0
lpips_loss = 0
