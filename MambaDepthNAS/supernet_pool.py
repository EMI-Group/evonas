import time
import torch
from collections import OrderedDict
import torch.distributed as dist

class ArchitecturePool:
    def __init__(self, pool_size=3):
        self.pool_size = pool_size
        self.pool = []  # List to store architectures
        self.scores = []  # List to store architectures
        self.timestamps = []  # List to store timestamps for each architecture

    def add(self, architecture):
        self.pool.append(architecture)

    def __iter__(self):
        # 让这个类可迭代，返回架构的迭代器
        return iter(self.pool)

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index):
        return self.pool[index]
    
    def cal_arch_dis(self, arch1, arch2):
        # 计算架构之间的距离，参考你之前的逻辑
        dis = 28
        n_layers = 28
        for i in range(n_layers):
            if arch1[i] == arch2[i]:
                dis -= 1
        dis = dis / 28
        return dis
    
    def add_architecture_time(self, new_arch):
        if len(self.pool) < self.pool_size:
            # 如果池未满，直接添加架构和时间戳
            self.pool.append(new_arch)
            self.timestamps.append(time.time())  # 记录当前时间戳
        else:
            # 如果池已满，找到池中**最早添加的架构**（即最老的架构）
            oldest_idx = self.timestamps.index(min(self.timestamps))  # 找到时间最早的架构的索引

            # 替换池中指定的架构及其对应的时间戳
            self.pool[oldest_idx] = new_arch
            self.timestamps[oldest_idx] = time.time()  # 更新时间戳

    def add_architecture(self, new_arch):
        if len(self.pool) < self.pool_size:
            # 如果池未满，直接添加架构和对应的分数
            self.pool.append(new_arch)

            self.timestamps.append(time.time())  # 记录当前时间戳
        else:
            # 如果池已满，寻找最相似的架构以替换
            min_dis = float('inf')
            candidates = []  # 候选架构的索引列表

            # 计算新架构与池中每个架构的距离，找到距离最小的架构
            for i, arch in enumerate(self.pool):
                dis = self.cal_arch_dis(new_arch, arch)
                if dis < min_dis:
                    min_dis = dis
                    candidates = [i]  # 更新候选索引
                elif dis == min_dis:
                    candidates.append(i)  # 追加与新架构距离相等的索引

            # 若多个架构与新架构距离相同，选择最早添加的架构
            if len(candidates) > 1:
                earliest_idx = min(candidates, key=lambda x: self.timestamps[x])
            else:
                earliest_idx = candidates[0]

            # 替换池中指定的架构及其对应的 SynFlow 分数和时间戳
            self.pool[earliest_idx] = new_arch

            self.timestamps[earliest_idx] = time.time()  # 更新时间戳

    def print_pool(self):
        print("Current architectures in the pool:")
        for i, arch in enumerate(self.pool):
            print(f"Index {i}: {arch}")


def get_grad_norm_arr(net, data, split_data=1, skip_grad=False):
    net.zero_grad()  # 清除所有梯度
    # N = inputs.shape[0]  # 数据的 batch 大小

    # 对输入数据进行分批次计算（避免显存占用过高）
    # for sp in range(split_data):  # TODO
    #     st = sp * N // split_data
    #     en = (sp + 1) * N // split_data

    # 使用指定的采样的网络结构
    losses, _ = net(**data)
    loss, _ = parse_losses(losses)

    params = [p for p in net.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

    total_grad_norm = 0.0
    total_grad_norm = sum(g.norm().item() for g in grads if g is not None)

    total_grad_norm = reduce_tensor(total_grad_norm)

    return total_grad_norm

def reduce_tensor(tensor, average=True, op=dist.ReduceOp.SUM):
    """
    在分布式环境下同步任意 GPU 上的张量，并可选择是否取平均.
    
    Args:
        tensor (float or torch.Tensor): 要同步的标量或张量
        average (bool): 是否在 all_reduce 后除以 world_size
        op: 规约操作 (默认 SUM，可选 MAX, MIN, PRODUCT)
    
    Returns:
        torch.Tensor: 同步后的张量（在所有进程上相同）
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32, device='cuda')

    dist.all_reduce(tensor, op=op)
    if average and op == dist.ReduceOp.SUM:
        tensor /= world_size

    return tensor


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    loss = reduce_tensor(loss)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars