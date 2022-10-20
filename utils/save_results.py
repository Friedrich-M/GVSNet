import os
from pathlib import Path
import imageio
import cv2 as cv
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from PIL import Image

from .semantics_palettes import get_palette
from .semantics_palettes import get_num_classes


def animate_files(input_files, output_file_name):
    # cast to string: incase we pass a PosixPath object 
    output_file_name = str(output_file_name)
    if os.path.exists(output_file_name):
        os.system(f'rm {output_file_name}')
    frames = [np.array(imageio.imread(f), dtype=np.uint8) for f in input_files]
    imageio.mimsave(output_file_name, frames)

def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype) 
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)]) # 将整数转换为二进制

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8) 
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i + 1  # let's give 0 a color
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
        
class SaveSemantics:    
    '''
    Currently supports the following datasets
    ['carla', 'vkitti', 'cityscapes']
    I add lhq dataset
    '''

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name # 'carla', 'vkitti', 'cityscapes', 'lhq'
        self.num_classes = get_num_classes(dataset_name)
        self.pallete = get_palette(dataset_name)

    def __call__(self, input_seg, file_name):
        input_seg = input_seg.squeeze()
        assert input_seg.ndimension(
        ) == 2, 'input segmentation should be either [H, W] or [1, H, W]'
        self.save_lable(input_seg, file_name)

    def to_color(self, input_seg):
        assert self.num_classes > input_seg.max(), 'Segmentaion mask > num_classes'
        input_seg = input_seg.int().squeeze().numpy()
        seg_mask = np.asarray(input_seg, dtype=np.uint8)
        pil_im = Image.fromarray(seg_mask, mode="P")
        pallette_ = []
        for v in self.pallete.values():
            pallette_.extend(v)
        for _i in range(len(self.pallete.keys()), 256):
            pallette_.extend([0, 0, 0])
        pil_im.putpalette(pallette_)
        pil_np = np.asarray(pil_im.convert('RGB'), dtype=np.uint8)
        return pil_np

    def save_lable(self, input_seg, file_name):
        col_img = self.to_color(input_seg)        
        pil_im = Image.fromarray(col_img).convert('PNG')
        pil_im.save(file_name)


class SaveResults:
    def __init__(self, output_path, dataset='carla'):
        self.output_path = output_path
        self.dataset = dataset
        self.write_semantics = SaveSemantics(dataset)

    def write_color(self, img_array, file_name):
        file_name = str(file_name)
        img_array = img_array.transpose(1, 2, 0)*255
        # RGB2BGR
        img_array = img_array[..., [2, 1, 0]]
        cv.imwrite(file_name, img_array)

    def write_depth(self, depth_tensor, file_name, normalize=True):
        raise NotImplementedError

    def _adjust_range(self, input_tensor):
        return (input_tensor+1)/2.0

    def save_color_imgs(self, img_tensor, itr, adjust_range=True):
        img_shape = 'img_tensor shape should be [num_of_cams, batch_size, 3, height, width]'
        assert img_tensor.ndimension() == 5, img_shape
        img_list = [im.squeeze(1) for im in torch.split(
            img_tensor, dim=1, split_size_or_sections=1)]
        if adjust_range:
            img_list = [self._adjust_range(im) for im in img_list]
        img_list = [im.to('cpu').numpy() for im in img_list]
        for b, img in enumerate(img_list):
            scene_folder = Path(self.output_path) / \
                f'scene_{str(b+itr).zfill(4)}'
            os.makedirs(scene_folder.__str__(), exist_ok=True)
            file_names = []
            for view in range(img.shape[0]):
                f_name = scene_folder / f'color_nv_{str(view).zfill(4)}.png'
                self.write_color(img[view], f_name)
                file_names.append(f_name.__str__())
            animate_files(file_names, scene_folder / f'animation_color_nv.gif')

    def __call__(self, result_dict, itr):
        '''
        This class assumes you are passing a dictionary of tensors with the following keys
        color_nv: [num_of_cams, batch_size, 3, height, width]
        disp_nv: [num_of_cams, 1, 3, height, width]
        sem_nv:  [num_of_cams, num_classes, 3, height, width]
        '''
        if 'color_nv' in result_dict.keys():
            self.save_color_imgs(result_dict['color_nv'], itr)
        # if 'disp_nv' in result_dict.keys():
        #     self.save_disp_imgs(result_dict['disp_nv'])
        # if 'sem_nv' in result_dict.keys():
        #     self.save_sem_imgs(result_dict['sem_nv'])
