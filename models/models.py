from os.path import join

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
from . import netvlad_region
import torchvision.transforms as transforms
import numpy as np


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


class RootNorm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return torch.sqrt(torch.div(input_data.abs(), input_data.abs().sum()))


def get_backend():
    pretrained = True

    enc = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    remove_layer_num = 2
    enc_dim = 512

    layers = list(enc.features.children())[:-remove_layer_num]

    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = False

    enc = nn.Sequential(*layers)
    return enc_dim, enc


def get_model(encoder, encoder_dim, opt, append_pca_layer=False):
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoder)

    net_vlad = netvlad_region.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2,
                                      region_size=(opt.regionSize, opt.regionSize),
                                      move_amount=(opt.moveAmount, opt.moveAmount),
                                      get_labels=False)

    nn_model.add_module('pool', net_vlad)

    if append_pca_layer:
        num_pcs = opt.num_pcs
        netvlad_output_dim = encoder_dim
        netvlad_output_dim *= opt.num_clusters

        pca_conv = nn.Conv2d(netvlad_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
        pca_conv.requires_grad = False
        nn_model.add_module(opt.use_pca, nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))

    return nn_model


def get_resume_ckpt(opt):
    if opt.use_pca is not None:
        ckpt_to_resume = join(opt.resume, 'checkpoints', 'model_' + opt.ckpt.lower() + '_' +
                              opt.use_pca + str(opt.num_pcs) + '.pth.tar')
        should_resume_pca = True
    else:
        should_resume_pca = False
        if opt.ckpt.lower() == 'latest':
            ckpt_to_resume = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            ckpt_to_resume = join(opt.resume, 'checkpoints', 'model_best.pth.tar')
        else:
            raise ValueError('--ckpt has to be either \'latest\' or \'best\'')

    return ckpt_to_resume, should_resume_pca


def get_pca_encoding(model, vlad_encoding, resume_ckpt):
    if '_WPCA' in resume_ckpt:
        pca_encoding = model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
    elif '_PCA' in resume_ckpt:
        pca_encoding = model.PCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
    else:
        raise ValueError('Unknown PCA type')
    return pca_encoding


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def calc_indices(regionSize, moveAmount):
    H = 480 / 16
    W = 640 / 16
    paddingSize = [0, 0]
    region_size = (int(regionSize), int(regionSize))
    move_amount = (moveAmount, moveAmount)

    Hout = (H + (2 * paddingSize[0]) - region_size[0]) / move_amount[0] + 1
    Wout = (W + (2 * paddingSize[1]) - region_size[1]) / move_amount[1] + 1

    Hout = int(Hout)
    Wout = int(Wout)

    numRegions = Hout * Wout

    k = 0
    All_indices = np.zeros((2, numRegions), dtype=int)
    for i in range(0, Hout):
        for j in range(0, Wout):
            All_indices[0, k] = j
            All_indices[1, k] = i
            k += 1
    return numRegions, Hout, Wout, All_indices
