import os
import math
import random
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
import csv


class LHQ(Dataset):
    def __init__(self, opts):
        super(LHQ, self).__init__()
        self.opts = opts
        assert os.path.exists(opts.data_path), f'path {opts.data_path} does not exist'
        # Transformations
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.ToPIL = transforms.Compose([transforms.ToPILImage()])       



    def __getitem__(self, index):
        if self.opts.mode=='train':
            sample = self.file_list[index]
            trg_cam, src_cam = random.sample(self.train_camera_suffix, 2) # 从_00 _01 _02 _03 _04中任取两个作为target和source相机视角
            cam_group = Path(sample).parent.parent.stem # 获取相机的类别 如HorizontalCameras
            src_file = sample.replace(cam_group, cam_group+src_cam) # 加上_00等后缀
            trg_file = sample.replace(cam_group, cam_group+trg_cam)
        else:
            src_file, trg_file = self.file_list[index][0], self.file_list[index][1]
        input_img = self._read_rgb(src_file) # 输入视角的图片
        target_img = self._read_rgb(trg_file) # 目标视角的图片 
        k_matrix = self._carla_k_matrix(height=self.height, width=self.width, fov=90) # 得到相机内参矩阵
        input_disp = self._read_disp(src_file.replace('rgb', 'depth'), k_matrix) # 输入视角的视差图
        target_disp = self._read_disp(trg_file.replace('rgb', 'depth'), k_matrix) # 输出视角的视差图
        input_seg = self._read_seg(src_file.replace('rgb', 'semantic_segmentation')) # 输入视角的分割图
        target_seg = self._read_seg(trg_file.replace('rgb', 'semantic_segmentation')) # 输出视角的分割图
        r_mat, t_vec = self._get_rel_pose(src_file, trg_file) # 得到相机的外参矩阵（旋转矩阵和平移矩阵）
        data_dict = {} # 以字典的形式存储
        data_dict['input_img'] = input_img
        data_dict['input_seg'] = input_seg
        data_dict['input_disp'] = input_disp
        data_dict['target_img'] = target_img
        data_dict['target_seg'] = target_seg
        data_dict['target_disp'] = target_disp
        data_dict['k_matrix'] = k_matrix
        data_dict['t_vec'] = t_vec
        data_dict['r_mat'] = r_mat
        data_dict['stereo_baseline'] = torch.Tensor([self.stereo_baseline])
        # Load style image, if passed, else the input will serve as style
        data_dict['style_img'] = input_img.clone()
        data_dict = {k: v.float()
                     for k, v in data_dict.items() if not (k is None)}
        return data_dict
        
    def _get_rel_pose(self, src_file, trg_file):
        cam_src = Path(src_file).parent.parent.stem
        cam_trg = Path(trg_file).parent.parent.stem
        src_idx, trg_idx = int(cam_src[-2:]), int(cam_trg[-2:])
        if cam_src.startswith('ForwardCameras'):
            x, y = 0, 0
            z = (src_idx - trg_idx)*self.stereo_baseline
        elif cam_src.startswith('HorizontalCameras'):
            y, z = 0, 0
            x = (src_idx - trg_idx)*self.stereo_baseline
        elif cam_src.startswith('SideCameras'):
            y, z = 0, 0
            x = (trg_idx - src_idx)*self.stereo_baseline
        else:
            assert False, f'unknown camera identifier {cam_src}'

        t_vec = torch.FloatTensor([x, y, z]).view(3, 1)
        r_mat = torch.eye(3).float()
        return r_mat, t_vec

    def _read_depth(self, depth_path):
        img = np.asarray(Image.open(depth_path), dtype=np.uint8)
        img = img.astype(np.float64)[:,:,:3]
        normalized_depth = np.dot(img, [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        normalized_depth = torch.from_numpy(normalized_depth * 1000.0)
        return normalized_depth

    def _read_disp(self, depth_path, k_matrix):
        depth_img = self._read_depth(depth_path).squeeze() # 读入深度图
        disp_img = self.stereo_baseline * \
            (k_matrix[0, 0]).view(1, 1) / (depth_img.clamp(min=1e-06)).squeeze() # 从深度图转换得到视差图
        h, w = disp_img.shape[:2]
        disp_img = disp_img.view(1, 1, h, w)
        disp_img = F.interpolate(disp_img, size=(self.height, self.width), 
                                mode='bilinear', align_corners=False) # 双线性插值将视察图resize到256*256大小
        disp_img = disp_img.view(1, self.height, self.width)
        return disp_img

    def __len__(self):
        return len(self.file_list)

    def label_to_one_hot(self, input_seg, num_classes=182):
        # 把seg转换为独热编码的格式
        assert input_seg.max() < num_classes, f'Num classes == {input_seg.max()} exceeds {num_classes}' # 保证seg标签的最大值不超过label的类别数
        b, _, h, w = input_seg.shape
        lables = torch.zeros(b, num_classes, h, w).float() # b是图片数，num_classes是label数，h是高，w是宽
        labels = lables.scatter_(dim=1, index=input_seg.long(), value=1.0)
        labels = labels.to(input_seg.device)
        return labels

    def _read_seg(self, semantics_path):
        seg = cv.imread(semantics_path, cv.IMREAD_ANYCOLOR |
                        cv.IMREAD_ANYDEPTH)
        seg = np.asarray(seg, dtype=np.uint8)
        seg = torch.from_numpy(seg[..., 2]).float().squeeze()
        h, w = seg.shape
        seg = F.interpolate(seg.view(1, 1, h, w), size=(self.height, self.width),
                            mode='nearest')
        # Change semantic labels to one-hot vectors
        seg = self.label_to_one_hot(seg, self.opts.num_classes).squeeze(0)
        return seg

    def _carla_k_matrix(self, fov=90.0, height=256, width=256):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / \
            (2.0 * math.tan(fov * math.pi / 360.0))
        return torch.from_numpy(k)

    def _read_rgb(self, img_path):
        img = io.imread(str(img_path))
        img = img[:, :, :3]
        img = cv.resize(img, (self.width, self.height)) / 255.0 # normalize to [0, 1]
        img = (2*img)-1.0 # normalize to [-1, 1]
        img_tensor = torch.from_numpy(img).transpose(
            2, 1).transpose(1, 0).float() # HWC -> CHW
        return img_tensor
