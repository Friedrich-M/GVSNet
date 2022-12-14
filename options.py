import argparse
from utils import get_current_time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu_id', type=int, default=0, help='pass -1 for cpu') # gpu的编号，-1表示使用cpu
arg_parser.add_argument('--dataset', type=str,
                        default='carla_samples',
                        help='dataset type: choose from carla_samples, carla, vkitti, cityscapes') # 数据集的类型
arg_parser.add_argument('--height', type=int, default=256) # 图像的高度
arg_parser.add_argument('--width', type=int, default=256) # 图像的宽度
arg_parser.add_argument('--batch_size', type=int, default=1) # batch的大小
arg_parser.add_argument('--lr', type=float, default=0.0004) # 学习率

arg_parser.add_argument('--slurm', action='store_true') # 是否使用slurm
arg_parser.add_argument('--port', type=int, default='7007') # 端口号
arg_parser.add_argument('--data_path', type=str, default='/data/teddy/carla', help='folder containing the dataset') # 数据集的路径
arg_parser.add_argument('--num_epochs', type=int,
                        default=30, help='number of training epochs') # 训练的epoch数
arg_parser.add_argument('--logging_path', type=str,
                        default=f'./logging/{get_current_time()}', help='path for saving training logs') # 日志的保存路径
arg_parser.add_argument('--image_log_interval', default=1000, type=int, help='number of iterations for saving intemediate outputs') # 保存中间结果的间隔
arg_parser.add_argument('--pre_trained_gvsnet', type=str, default=f'./pre_trained_models/carla/gvsnet_model.pt',
                        help='path for pre_trained_gvsnet') # 预训练的gvsnet模型的路径
arg_parser.add_argument('--pre_trained_sun', type=str, default=f'./pre_trained_models/carla/sun_model.pt',
                        help='path for sun model| useful when training LTN and ADN models') # 预训练的sun模型的路径

arg_parser.add_argument('--num_classes', type=int, default=183) # label标签的类别数
arg_parser.add_argument('--embedding_size', type=int, default=183, 
                        help='when # of semantic classes is large SUN and LTD will be fed with lower dimensoinal embedding of semantics') # embedding的维度
arg_parser.add_argument('--stereo_baseline', type=float, default=0.24, help='assumed baseline for converting depth to disparity') # 立体视觉的基线
arg_parser.add_argument('--style_path', type=str, default='', help='if given the this file will be used as style image') # 风格图片的路径
arg_parser.add_argument('--use_instance_mask', action='store_true', help='is paased, instance mask will be assuned to be present') # 是否使用实例分割的mask
arg_parser.add_argument('--mpi_encoder_features', type=int, default=96, help='this controls number feature channels at the output of the base encoder-decoder network') # mpi的编码器的特征数
arg_parser.add_argument('--mode', type=str, default='test', help='choose between [train, test, demo]') # 模式选择，训练，测试，演示

arg_parser.add_argument('--num_layers', type=int, default=3) # lift semantic的层数
arg_parser.add_argument('--feats_per_layer', type=int, default=20) 
arg_parser.add_argument('--num_planes', type=int, default=32) # alpha mpi的平面数
arg_parser.add_argument('--near_plane', type=int, default=1.5, help='nearest plane: 1.5 for carla') # 最近的平面
arg_parser.add_argument('--far_plane', type=int, default=20000, help='far plane: 20000 for carla') # 最远的平面

######### Arguments for SPADE
arg_parser.add_argument('--d_step_per_g', type=int, default=1, help='num of d updates for each g update') # 每次g更新的d的更新次数
arg_parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)') # 裁剪的大小
arg_parser.add_argument('--spade_k_size', type=int,default=3) # spade的卷积核大小
arg_parser.add_argument('--num_D', type=int, default=3) # 判别器的个数
arg_parser.add_argument('--output_nc', type=int, default=3) # 输出的通道数
arg_parser.add_argument('--n_layers_D', type=int, default=4) # 判别器的层数

arg_parser.add_argument('--contain_dontcare_label', action='store_true') # 是否包含dontcare的label
arg_parser.add_argument('--no_instance', default=True, type=bool) # 是否使用实例分割
arg_parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3', help='instance normalization or batch normalization') # 归一化方法
arg_parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization') 
arg_parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')

## Generator settings
arg_parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer') # 生成器的第一层的卷积核数
arg_parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]') # 初始化方法
arg_parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution') # 初始化的方差
arg_parser.add_argument('--z_dim', type=int, default=256, help="dimension of the latent z vector") # 隐变量的维度
## Discriminator setting
arg_parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer') # 判别器的第一层的卷积核数
arg_parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss') # 特征匹配的权重
arg_parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss') # vgg的权重
arg_parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss') # 是否使用gan特征匹配的损失
arg_parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss') # 是否使用vgg特征匹配的损失
arg_parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)') # gan的模式
arg_parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)') # 判别器的模式
arg_parser.add_argument('--lambda_kld', type=float, default=0.001) # kld的权重
arg_parser.add_argument('--num_upsampling_layers',
                    choices=('normal', 'more', 'most'), default='normal',
                    help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator") # 上采样的层数
arg_parser.add_argument('--num_out_channels', default=3, type=int) # 输出的通道数
arg_parser.add_argument('--use_vae', action='store_false') # TODO: This should be updated
arg_parser.add_argument('--aspect_ratio', default=1, type=int) # 宽高比
arg_parser.add_argument('--gen_lr', default=0.0001, type=float) # 生成器的学习率
arg_parser.add_argument('--disc_lr', default=0.0004, type=float) # 判别器的学习率

## Training options
arg_parser.add_argument('--disparity_weight', default=0.1, type=float, 
                        help='for carla=0.1, for other set to 0.5') # 深度图的权重

