import os
import sys
import torch
import random
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch import distributed as dist

from options import arg_parser
from models import SUNModel
from data import get_dataset
from utils import lr_func
from utils import Logger
from utils import dummy_progress_bar

# 设置随机种子
torch.random.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

arg_parser.add_argument('--ngpu', type=int, default=1)  # gpu个数
arg_parser.add_argument('--local_rank', type=int, default=0)  # gpu编号
opts = arg_parser.parse_args()

# Initialize process group
if opts.slurm:
    def init_process_group():
        # os.environ['NCCL_IB_DISABLE'] = '1'
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        rank = int(os.getenv('RANK', 0))
        world_size = int(os.getenv('WORLD_SIZE', 1))
        num_nodes = int(os.getenv('SLURM_NNODES', 1))
        print(f'ank {rank} | Local rank {local_rank} | world size {world_size}')
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        return rank, world_size, num_nodes
    rank, world_size, num_nodes = init_process_group()
    opts.__dict__['local_rank'] = rank
    opts.__dict__['world_size'] = world_size
    device = f'cuda:{opts.local_rank}'
    torch.cuda.set_device(opts.local_rank)
else:
    device = f'cuda:{opts.local_rank}'  # 'cuda:0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{opts.port}'
    torch.cuda.set_device(opts.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=opts.ngpu,
        rank=opts.local_rank,
    )
# Prepare logging path
if opts.local_rank == 0:
    print(f'Find tensorboard at {opts.logging_path}')
    os.makedirs(opts.logging_path, exist_ok=True)
    model_path = os.path.join(opts.logging_path, 'models') # 模型保存路径
    image_path = os.path.join(opts.logging_path, 'images') # 图片保存路径
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

# Create model
model = SUNModel(opts) # 创建模型
model = model.to(device) # 模型放到GPU上

dataset = get_dataset(opts.dataset)(opts) # 获取数据集
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=opts.world_size if opts.slurm else opts.ngpu,
    rank=opts.local_rank,
    shuffle=True)  # 分布式采样器（多gpu训练）

data_loader = DataLoader(dataset=dataset,
                         batch_size=opts.batch_size,
                         sampler=sampler,
                         drop_last=True,
                         num_workers=opts.batch_size,
                         pin_memory=True,
                         ) # 数据加载器
# Adam优化器
optimizer = torch.optim.Adam(
    model.parameters(), lr=opts.lr, betas=(0.9, 0.999))
if opts.slurm:
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device],) # 模型太大而无法容纳在单个GPU上，则必须使用 model parallel 将其拆分到多个GPU中。 DistributedDataParallel与模型并行工作
else:
    model_ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,) # 分布式模型

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lr_func(opts.num_epochs), last_epoch=-1) # 学习率调整器


logger = Logger(opts.logging_path) if opts. local_rank == 0 else None # 日志记录器
for epoch in range(opts.num_epochs):
    optimizer.zero_grad() # 梯度清零
    sampler.set_epoch(epoch) # 在分布式模式中，需要在每个epoch获取到数据之前，调用set_epoch函数，保证可以在多个epoch上获取顺序被打乱的数据，即shuffle=True
    if logger:
        logger.save_model(model, epoch) # 保存模型
    with tqdm.tqdm(total=len(data_loader)) if opts.local_rank == 0 else dummy_progress_bar() as progress_bar: 
        for itr, data in enumerate(data_loader):
            data = {k: v.float().to(device) for k, v in data.items()} 
            loss_dict, semantics_nv, disp_iv = model_ddp(data, mode='training')
            loss = sum([v for k, v in loss_dict.items()]) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 梯度清零
            # Logging loss, predicted images etc
            if logger:
                logger.log_scalar(loss_dict) 
                if itr % opts.image_log_interval == 0:
                    # pred_sem_nv = semantics_nv.data.squeeze().cpu() # 预测的语义图
                    # target_sem_nv = data['target_seg'].squeeze().cpu() # 真实的语义图
                    # input_sem = data['input_seg'].squeeze().cpu() # 输入的语义图
                    pred_sem_nv = semantics_nv.data.cpu() # 预测的语义图
                    target_sem_nv = data['target_seg'].cpu() # 真实的语义图
                    input_sem = data['input_seg'].cpu() # 输入的语义图
                    logger.save_semantics({'sem_pred_novel_v': pred_sem_nv[0]}) # 保存预测的语义图
                    logger.save_semantics({'sem_gt_novel_v': target_sem_nv[0]}) # 保存真实的语义图
                    logger.save_semantics({'sem_gt_input_v': input_sem[0]}) # 保存输入的语义图
                    logger.save_images(
                        {'rgb_gt_input_v': data['input_img'][0].cpu()}) # 保存输入的RGB图
                    logger.save_depth(
                        {'disp_gt_input_v': data['input_disp'][0].cpu()}) # 保存输入的深度图
                    logger.save_depth({'disp_pred_input_v': disp_iv[0].cpu()}) # 保存预测的深度图
            if progress_bar:
                progress_bar.set_postfix(disp_loss=loss_dict['disp_loss'].item(),
                                         sem_loss=loss_dict['semantics_loss'].item(
                ),
                    lr=optimizer.param_groups[0]['lr']) # 设置进度条
                progress_bar.update(1) # 更新进度条
            if logger:
                logger.step() # 记录步数
