import torch
import torch.backends.cudnn as cudnn

import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import argparse
import numpy as np
import time
from networks.model import MambaDepth
from ptflops import get_model_complexity_info

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='MambaDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='MambaDepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='VSSD_final')
parser.add_argument('--decoder',                    type=str,  help='type of decoder,', default='mamba')
parser.add_argument('--neck',                    type=str,  help='type of module between encoder and decoder,', default='sp')

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
# parser.add_argument('--input_height',              type=int,   help='input height', default=352)
# parser.add_argument('--input_width',               type=int,   help='input width',  default=1216)
parser.add_argument('--width_multiplier',            type=float,  default=1.0)




if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


@torch.no_grad()
def latency_test(model, input_shape=(1,3,224,224), device='cuda', warmup=50, repeat=100):
    model.eval()
    input_tensor = torch.randn(*input_shape).to(device)

    # warmup
    for _ in range(warmup):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    # measure
    torch.cuda.reset_peak_memory_stats()
    tic1 = time.time()
    for _ in range(repeat):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    tic2 = time.time()

    total_time = (tic2 - tic1)
    avg_latency = total_time / repeat * 1000  # ms
    fps = repeat / total_time

    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"[Latency Test]")
    print(f"Input shape: {input_shape}")
    print(f"Average latency: {avg_latency:.3f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Max memory allocated: {max_mem:.2f} MB")

    return avg_latency, fps

def main_worker(args):

    if args.dataset == 'kitti':
        input_shape = (1,3,352,1216)
        args.input_height = 352
        args.input_width = 1216
    elif args.dataset == 'nyu':
        input_shape = (1,3,480,640)
        args.input_height = 480
        args.input_width = 640

    config= {}
    config['mlp_ratio'] = [2.0, 3.5, 3.0, 3.0]
    config['d_state'] = [64, 48, 64, -1]
    config['ssd_expand'] = [2, 3, 4, -1]
    config['depth'] = [1, 2, 7, 2]

    # MambaDepth model
    if args.decoder == 'mamba':
        Model = MambaDepth
    elif args.decoder == 'crfs':
        from networks.model import MambaDepth_Dec_CRFs
        Model = MambaDepth_Dec_CRFs
    elif args.decoder == 'idisc':
        from networks.model import MambaDepth_Dec_iDisc
        Model = MambaDepth_Dec_iDisc
    elif args.decoder == 'vmamba':
        from networks.model import MambaDepth_Dec_VMamba
        Model = MambaDepth_Dec_VMamba
    else:
        raise NotImplementedError
    model = Model(version=args.encoder, args=args, selected_config=config)
    model.train()

    # backbone_params = list(model.backbone.parameters())
    # ppm_params = list(model.PPM.parameters())
    # other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and not n.startswith("PPM.")]

    # def count_params(params):
    #     return sum(p.numel() for p in params)

    # total_params = count_params(model.parameters())
    # backbone_params_count = count_params(backbone_params)
    # ppm_params_count = count_params(ppm_params)
    # other_params_count = count_params(other_params)

    # print("="*50)
    # print(f"Total Parameters    : {total_params/1e6:.2f} M")
    # print(f"Backbone Parameters : {backbone_params_count/1e6:.2f} M")
    # print(f"PPM Parameters      : {ppm_params_count/1e6:.2f} M")
    # print(f"Other Parameters    : {other_params_count/1e6:.2f} M")
    # print("="*50)

    # with open(os.path.join('.', 'max_model.log'),'w') as f:
    #     f.write(str(model))
    # assert False,'print model'


    # '''show param'''
    # # 计算FLOPs 和 Params
    # '''https://github.com/MrYxJ/calculate-flops.pytorch?tab=readme-ov-file'''
    # print('*'*20, ' calflops ', '*'*20)
    # from calflops import calculate_flops
    # # from torchvision import models
    # import sys
    # with open('./FLOPs_Params.log', 'w') as f:
    #     original_stdout = sys.stdout
    #     sys.stdout = f
    #     # model = models.resnet50()
    #     # input_shape = (1, 3, 352, 1216)
    #     x = torch.randn(1,3,352,1216).cuda()
    #     flops, macs, params = calculate_flops(model=model, 
    #                                         # input_shape=input_shape,
    #                                         args=[x],
    #                                         output_as_string=True,
    #                                         output_precision=2)
    #     sys.stdout = original_stdout
    #     print("net FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))        
        #Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M 
    # assert False, 'over'

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_backbone_params = sum([np.prod(p.size()) for name, p in model.named_parameters() if 'backbone' in name])
    print("== Total number of backbone parameters: {}".format(num_backbone_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model.cuda()

    cudnn.benchmark = True

    # ===== Evaluation ======
    model.eval()
    latency_test(model, input_shape)

    macs, params = get_model_complexity_info(model, tuple(input_shape[1:]), as_strings=False, backend='pytorch', print_per_layer_stat=False)
    print(f'macs: {macs/1e9}G, params: {params/1e6}M')


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
