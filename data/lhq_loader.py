import os
from PIL import Image
from pathlib import Path
import cv2 as cv
from skimage import io
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset
import imageio
import sys


class LHQ(Dataset):
    def __init__(self, opts):
        super(LHQ, self).__init__()
        self.opts = opts
        # Transformations
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.ToPIL = transforms.Compose([transforms.ToPILImage()])

        self.height, self.width = self.opts.height, self.opts.width
        self.stereo_baseline = 0.58

        self.base_path = self.opts.data_path
        assert os.path.exists(
            self.base_path), f'path {self.base_path} does not exist'

        self.file_list = self.list_images()

    def list_images(self):
        path = Path(self.opts.data_path)
        file_dir = sorted(os.listdir(path))
        file_list = [os.path.join(path, f) for f in file_dir]
        return file_list

    def __getitem__(self, index):
        file_src = self.file_list[index]
        input_img_path = os.path.join(file_src, 'tgt_rgb.png')
        target_img_path = os.path.join(file_src, 'src_rgb.png')
        input_disp_path = os.path.join(file_src, 'tgt_disp.png')
        target_disp_path = os.path.join(file_src, 'src_disp.png')
        input_seg_path = os.path.join(file_src, 'tgt_seg.png')
        target_seg_path = os.path.join(file_src, 'src_seg.png')
        cam_ext_path = os.path.join(file_src, 'cam_ext.txt')
        cam_int_path = os.path.join(file_src, 'cam_int.txt')

        input_img = self._read_rgb(input_img_path)  # 输入视角的图片
        target_img = self._read_rgb(target_img_path)  # 目标视角的图片

        cam_int = np.loadtxt(cam_int_path, dtype=np.float32)  # 读取相机内参矩阵
        k_matrix = torch.from_numpy(cam_int)
        # k第一行*widht，第二行*height，第三行*1
        k_matrix[0, :] = k_matrix[0, :] * self.width
        k_matrix[1, :] = k_matrix[1, :] * self.height
        cam_ext = np.loadtxt(cam_ext_path, dtype=np.float32)  # 读取相机外参矩阵
        RT = torch.from_numpy(cam_ext)
        r_mat, t_vec = RT[:, :-1], RT[:, -1]  # 得到相机的外参矩阵（旋转矩阵和平移矩阵）

        input_disp = self._read_disp_(input_disp_path)  # 输入视角的视差图
        target_disp = self._read_disp_(target_disp_path)  # 输出视角的视差图
        input_seg = self._read_seg(input_seg_path)  # 输入视角的分割图
        target_seg = self._read_seg(target_seg_path)  # 输出视角的分割图

        data_dict = {}  # 以字典的形式存储
        data_dict['input_img'] = input_img
        data_dict['input_seg'] = input_seg
        data_dict['input_disp'] = input_disp
        data_dict['target_img'] = target_img
        data_dict['target_seg'] = target_seg
        data_dict['target_disp'] = target_disp
        data_dict['k_matrix'] = k_matrix
        data_dict['t_vec'] = t_vec
        data_dict['r_mat'] = r_mat
        # data_dict['stereo_baseline'] = torch.Tensor([self.stereo_baseline])
        # Load style image, if passed, else the input will serve as style
        data_dict['style_img'] = input_img.clone()
        data_dict = {k: v.float()
                     for k, v in data_dict.items() if not (k is None)}
        return data_dict

    def _read_disp_(self, disp_path):
        disp = imageio.imread(disp_path).astype(np.float32)[None]
        disp = disp - disp.min()
        disp = (disp / (disp.max()+1e-6)) * 1.0
        disp = np.maximum(disp, 0.05)
        disp = torch.from_numpy(disp)
        return disp

    def __len__(self):
        return len(self.file_list)

    def label_to_one_hot(self, input_seg, num_classes=183):
        # 把seg转换为独热编码的格式
        assert input_seg.max(
        ) < num_classes, f'Num classes == {input_seg.max()} exceeds {num_classes}'  # 保证seg标签的最大值不超过label的类别数
        b, _, h, w = input_seg.shape
        lables = torch.zeros(b, num_classes, h, w).float() # 生成一个全0的独热编码
        labels = lables.scatter_(dim=1, index=input_seg.long(), value=1.0) # 把seg的值赋给独热编码
        labels = labels.to(input_seg.device)
        return labels

    def _read_seg(self, semantics_path):
        seg = cv.imread(semantics_path, cv.IMREAD_ANYCOLOR |
                        cv.IMREAD_ANYDEPTH)  # 读入分割图
        seg = np.asarray(seg, dtype=np.uint8)
        seg = torch.from_numpy(seg).float().squeeze()
        h, w = seg.shape  # 256 * 256
        seg = F.interpolate(seg.view(1, 1, h, w), size=(self.height, self.width),
                            mode='nearest')  # 最近邻插值将seg图resize到256*256大小
        # Change semantic labels to one-hot vectors
        seg = self.label_to_one_hot(
            seg, self.opts.num_classes).squeeze(0)  # 把seg转换为独热编码的格式
        return seg

    def _read_rgb(self, img_path):
        img = io.imread(str(img_path))
        img = img[:, :, :3]
        img = cv.resize(img, (self.width, self.height)) / \
            255.0  # normalize to [0, 1]
        img = (2*img)-1.0  # normalize to [-1, 1]
        img_tensor = torch.from_numpy(img).transpose(
            2, 1).transpose(1, 0).float()  # HWC -> CHW
        return img_tensor
