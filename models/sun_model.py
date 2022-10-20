import sys
import torch

import torch.nn as nn
import torch.nn.functional as F
from .conv_network import ResBlock
from .conv_network import ConvBlock
from .conv_network import BaseEncoderDecoder

from .semantic_embedding import SemanticEmbedding
from .mpi import ComputeHomography
from .mpi import AlphaComposition
from .mpi import ApplyHomography
from .mpi import Alpha2Disp
from .mpi import ApplyAssociation


class MulLayerConvNetwork(torch.nn.Module):

    def __init__(self, opts):
        super(MulLayerConvNetwork, self).__init__()
        self.opts = opts
        input_channels = opts.num_classes  # 标签种类的个数 default=183
        num_planes = opts.num_planes  # mpi平面的个数 default=32
        # number feature channels at the output of the base encoder-decoder network default=96
        enc_features = opts.mpi_encoder_features
        self.input_channels = input_channels  # 183
        self.num_classes = opts.num_classes  # 183
        self.num_planes = num_planes  # 32
        self.out_seg_chans = self.opts.embedding_size  # 183
        self.discriptor_net = BaseEncoderDecoder(input_channels) # outchannels=96
        self.base_res_layers = nn.Sequential(
            *[ResBlock(enc_features, 3) for i in range(2)]) # 2个ResBlock kernel_size=3
        # we will re-use the input semantics
        # opts.num_layers = 3
        total_seg_channels = (self.opts.num_layers - 1) * \
            self.out_seg_chans  # （3-1）*183
        total_alpha_channels = num_planes  # 32
        # (k-1) extra seg layers each with 183 channels
        self.total_seg_channels = total_seg_channels
        # alpha with the original number of MPI layers m
        self.total_alpha_channels = total_alpha_channels
        self.total_beta_channels = num_planes * \
            self.opts.num_layers  # association matrix with k*m size
        total_output_channels = total_seg_channels + \
            total_alpha_channels + self.total_beta_channels  # the sum channels of seg, alpha and association
        self.blending_alpha_seg_beta_pred = nn.Sequential(ResBlock(enc_features, 3), # 1个ResBlock kernel_size=3
                                                          ResBlock(enc_features, 3), # 1个ResBlock kernel_size=3
                                                          nn.SyncBatchNorm(enc_features), # 为了多卡训练同步BN 96
                                                          ConvBlock(enc_features, total_output_channels // 2, 3, down_sample=False), # 96->total_output_channels // 2
                                                          nn.SyncBatchNorm(total_output_channels // 2), 
                                                          ConvBlock(total_output_channels // 2,
                                                                    total_output_channels, 3,
                                                                    down_sample=False,
                                                                    use_no_relu=True)) 

    def forward(self, input_sem):
        b, _, h, w = input_sem.shape #  b=1, 183, 256, 256 输入分割图的shape
        feats_0 = self.discriptor_net(input_sem) # 1, 96, 256, 256
        feats_1 = self.base_res_layers(feats_0) # 1, 96, 256, 256
        alpha_and_seg_beta = self.blending_alpha_seg_beta_pred(feats_1) # 1, 32+183+32*3, 256, 256
        alphas = alpha_and_seg_beta[:, -self.total_alpha_channels:, :, :]
        seg = alpha_and_seg_beta[:, self.total_beta_channels:
                                 self.total_beta_channels + self.total_seg_channels, :, :]
        beta = alpha_and_seg_beta[:, :self.total_beta_channels, :, :]
        alpha = alphas.view(b, self.num_planes, 1, h, w)
        seg = seg.view(b, (self.opts.num_layers - 1), self.out_seg_chans, h, w)
        beta = beta.view(b, self.num_planes, self.opts.num_layers, h, w)
        return alpha, seg, beta


class SUNModel(torch.nn.Module):
    '''
        A wrapper class for predicting MPI and doing rendering
    '''

    def __init__(self, opts):
        super(SUNModel, self).__init__()
        self.opts = opts
        self.conv_net = MulLayerConvNetwork(opts)
        self.compute_homography = ComputeHomography(opts)
        self.alpha_composition = AlphaComposition()
        self.apply_homography = ApplyHomography()
        self.alpha_to_disp = Alpha2Disp(opts)
        self.apply_association = ApplyAssociation(opts.num_layers)
        if not (opts.embedding_size == opts.num_classes):
            self.semantic_embedding = SemanticEmbedding(num_classes=opts.num_classes,
                                                        embedding_size=opts.embedding_size)

    def forward(self, input_data, mode='inference'):
        if mode == 'inference':  # inference mode
            with torch.no_grad():
                # For the representation, we use the input 2D semantic as the first layer and predict the remaining 2 layers using the sun model
                scene_representation = self._infere_scene_repr(
                    input_data)  # scene_representation = (alpha, seg, beta)
            return scene_representation
        elif mode == 'training':
            target_sem = input_data['target_seg']  # target_sem = (b, 1, h, w)
            seg_mul_layer, alpha, associations = self._infere_scene_repr(
                input_data)
            semantics_nv = self._render_nv_semantics(
                input_data, seg_mul_layer, alpha, associations)
            semantics_loss = self.compute_semantics_loss(
                semantics_nv, target_sem)
            if 'input_disp' in input_data.keys():
                disp_iv = self.alpha_to_disp(
                    alpha, input_data['k_matrix'], self.opts.stereo_baseline, novel_view=False)
                disp_loss = F.l1_loss(disp_iv, input_data['input_disp'])
            sun_loss = {'disp_loss': self.opts.disparity_weight * disp_loss,
                        'semantics_loss': semantics_loss}
            return sun_loss, semantics_nv.data, disp_iv.data

    def _infere_scene_repr(self, input_data):
        input_seg = input_data['input_seg']  # 输入的分割图
        encoding_needed = not (self.opts.num_classes ==
                               self.opts.embedding_size)  # 判断是否需要encoding
        input_seg_ = self.semantic_embedding.encode(
            input_seg) if encoding_needed else input_seg  # 如果num_classes小于embedding_size, 则需要encoding
        # Compute MPI alpha, multi layer semantics and layer to plane associations
        alpha, seg_mul_layer, associations = self.conv_net(input_seg_)
        # Append the input semantics to the multi layer semantics
        seg_mul_layer = F.softmax(seg_mul_layer, dim=2)  # softmax归一化
        seg_mul_layer = torch.cat(
            [input_seg_.unsqueeze(1), seg_mul_layer], dim=1)  # 将输入的分割图与多层分割图拼接
        alpha = torch.sigmoid(torch.clamp(
            alpha, min=-100, max=100))  # sigmoid归一化alpha
        associations = F.softmax(associations, dim=2)  # softmax归一化
        return seg_mul_layer, alpha, associations

    def _render_nv_semantics(self, input_data, layered_sem, mpi_alpha, associations):
        mpi_sem = self.apply_association(
            layered_sem, input_associations=associations)  # mpi semantices
        h_mats = self.compute_homography(
            kmats=input_data['k_matrix'], r_mats=input_data['r_mat'], t_vecs=input_data['t_vec'])
        mpi_sem_nv, grid = self.apply_homography(
            h_matrix=h_mats, src_img=mpi_sem, grid=None)
        mpi_alpha_nv, _ = self.apply_homography(
            h_matrix=h_mats, src_img=mpi_alpha, grid=grid)
        sem_nv = self.alpha_composition(
            src_imgs=mpi_sem_nv, alpha_imgs=mpi_alpha_nv)
        if not (self.opts.num_classes == self.opts.embedding_size):
            sem_nv = self.semantic_embedding.decode(sem_nv)
        return sem_nv

    def compute_semantics_loss(self, pred_semantics, target_semantics):
        _, target_seg = target_semantics.max(dim=1)
        return F.cross_entropy(pred_semantics, target_seg, ignore_index=self.opts.num_classes)
