# ----------------------------------------
# 数据集 & 路径配置（KITTI）
# ----------------------------------------
dataset = 'nyu'
data_path = '/data/dataset/NYU_Depth_V2/sync/'
gt_path = '/data/dataset/NYU_Depth_V2/sync/'
filenames_file = './train_test_inputs/nyudepthv2_train_files_with_gt.txt'
input_height = 480
input_width = 640

# ----------------------------------------
# 测试/评估配置
# ----------------------------------------
data_path_eval = '/data/dataset/NYU_Depth_V2/test/'
gt_path_eval = '/data/dataset/NYU_Depth_V2/test/'
filenames_file_eval = './train_test_inputs/nyudepthv2_test_files_with_gt.txt'
min_depth_eval = 0.001
max_depth_eval = 10

# ----------------------------------------
# 深度范围配置
# ----------------------------------------
min_depth = 0.001
max_depth = 10

# ----------------------------------------
# 数据增强
# ----------------------------------------
do_random_rotate = True
degree = 1.0
do_kb_crop = False
garg_crop = False
eigen_crop = True
random_crop = False
random_translate = False
translate_prob = 0.2
max_translation = 100
aug = True

# ----------------------------------------
# 分布式训练与系统设置
# ----------------------------------------
gpu = None
distributed = True
workers = 16
clip_grad = 0.1
prefetch = False
use_shared_dict = False
shared_dict = None
use_amp = False
print_losses = False

# ----------------------------------------
# 日志 & 保存
# ----------------------------------------
save_dir = './depth_anything_finetune'
project = 'ZoeDepth'
tags = ''
notes = ''
root = '.'
uid = None
validate_every = 0.25
log_images_every = 0.1
name = 'ZoeDepth'
version_name = 'v1'

# ----------------------------------------
# 模型结构 & 参数
# ----------------------------------------
model = 'zoedepth'
img_size = [392, 518]
midas_model_type = 'DPT_BEiT_L_384'
inverse_midas = False
train_midas = False
use_pretrained_midas = False
pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'

# ----------------------------------------
# Bin & Attractor 深度建模配置
# ----------------------------------------
n_bins = 64
bin_embedding_dim = 128
bin_centers_type = 'softplus'

n_attractors = [16, 8, 4, 1]
attractor_alpha = 1000
attractor_gamma = 2
attractor_kind = 'mean'
attractor_type = 'inv'

min_temp = 0.0212
max_temp = 50.0
output_distribution = 'logbinomial'
memory_efficient = True

avoid_boundary = False
max_depth_diff = 10
min_depth_diff = -10
