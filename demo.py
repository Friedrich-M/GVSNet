import os
import torch
import tqdm
from torch.utils.data import DataLoader

from options import arg_parser
from models import GVSNet
from data import get_dataset
from utils import SaveResults
from utils import get_current_time
from utils import get_cam_poses
from utils import convert_model


arg_parser.add_argument('--movement_type', default='circle', choices=['circle', 'horizontal', 'forward'], 
                        help='camera movement type: ') # circle, horizontal, forward
arg_parser.add_argument('--output_path', type=str, default=f'./output/{get_current_time()}', 
                        help='path for saving results') 



opts = arg_parser.parse_args()
device = f'cuda:{opts.gpu_id}' if opts.gpu_id>-1 else 'cpu'

# prapare folder for results 
os.makedirs(opts.output_path, exist_ok=True)

gvs_net = GVSNet(opts)
# torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
# 当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合；
# strict=False,与训练权重中与新构建网络中匹配层的键值就进行使用，没有的就默认初始化。
gvs_net.load_state_dict(torch.load(opts.pre_trained_gvsnet), strict=True)
gvs_net.to(device)
# eval()时，框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
# 不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大
gvs_net.eval() 

if device=='cpu':
    gvs_net = convert_model(gvs_net)
# 生成数据集
dataset = DataLoader(get_dataset(opts.dataset)(opts),
                     batch_size=1, shuffle=False)

saver_results = SaveResults(opts.output_path, opts.dataset)

for itr, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
    data = {k:v.to(device) for k,v in data.items()}
    # Let's get a list of camera poses
    # modify get_cam_poses function if you need specific camera movement
    data['t_vec'], data['r_mat'] = get_cam_poses(opts.movement_type, b_size=opts.batch_size)
    data['t_vec'] = [t.to(device) for t in data['t_vec']]
    data['r_mat'] = [r.to(device) for r in data['r_mat']]
    # Render the scene from the chosen camera poses
    results_dict = gvs_net.render_multiple_cams(data)
    saver_results(results_dict, itr)
print(f'Find results at {opts.output_path}')
