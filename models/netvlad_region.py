import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np


# will edit this file to create the region netvlad layer (this file will replace the original netvlad.py in main.py)

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, use_faiss=True,
                 region_size=(5, 5), move_amount=(2, 2), get_labels=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        # noinspection PyArgumentList
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss
        self.region_size = region_size
        self.move_amount = move_amount
        self.get_labels = get_labels

    def init_params(self, clsts, traindescs):
        # TODO replace numpy ops with pytorch ops
        if not self.vladv2:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            if not self.use_faiss:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(traindescs)
                del traindescs
                dsSq = np.square(knn.kneighbors(clsts, 2)[1])
                del knn
            else:
                index = faiss.IndexFlatL2(traindescs.shape[1])
                # noinspection PyArgumentList
                index.add(traindescs)
                del traindescs
                # noinspection PyArgumentList
                dsSq = np.square(index.search(clsts, 2)[1])
                del index

            self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            # noinspection PyArgumentList
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C, H, W = x.shape  # reminder: N is batch size

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, H, W)
        soft_assign = F.softmax(soft_assign, dim=1)

        paddingSize = (0, 0)

        # numRegions = ((H - self.region_size[0] + self.move_amount[0])/self.move_amount[0])*((W - self.region_size[1] + self.move_amount[1])/self.move_amount[1])
        # numRegions = int(numRegions)

        Hout = (H + (2 * paddingSize[0]) - self.region_size[0]) / self.move_amount[0] + 1
        Wout = (W + (2 * paddingSize[1]) - self.region_size[1]) / self.move_amount[1] + 1

        Hout = int(Hout)
        Wout = int(Wout)

        # calculate residuals to each clusters
        # vlad = torch.zeros([N, self.num_clusters, C, Hout, Wout], dtype=x.dtype, layout=x.layout, device=x.device)
        store_residual = torch.zeros([N, self.num_clusters, C, H, W], dtype=x.dtype, layout=x.layout, device=x.device)
        for j in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x.unsqueeze(0).permute(1, 0, 2, 3, 4) - \
                       self.centroids[j:j + 1, :].expand(x.size(2), x.size(3), -1, -1).permute(2, 3, 0, 1).unsqueeze(0)

            residual *= soft_assign[:, j:j + 1, :].unsqueeze(2)  # residual should be size [N 64 C H W]
            store_residual[:, j:j + 1, :, :, :] = residual

        opR = nn.AvgPool2d(self.region_size,
                           self.move_amount)  # this will act as a normalized version of my original regionNetVLAD

        vlad_full = store_residual.view(N, self.num_clusters, C, -1)
        vlad_full = vlad_full.sum(dim=-1)

        store_residual = store_residual.view(N, -1, H, W)

        vladflattened = opR(store_residual)

        # labels are derived from the soft assignments per spatial element
        # for region labels, using the average assignment (TODO: better alternatives?)
        if self.get_labels:
            distMode = True  # computes distance from cc; perhaps corresponds with vladv2
            if distMode:
                soft_assign_regions = ((x[:, None, :, :, :] - self.centroids[None, :, :, None, None]) ** 2).sum(dim=2)
            soft_assign_regions = opR(soft_assign_regions)
            labels_regions = torch.argsort(soft_assign_regions, dim=1, descending=(not distMode))[:, :2, ...]

        vlad = vladflattened.view(N, self.num_clusters, C, Hout, Wout)

        vlad = F.normalize(vlad, p=2, dim=2)

        #      for j in range(C):
        # vlad[:, :, j, :, :] = opR(store_residual[:, :, j, :, :])
        #          vlad[:, :, j, :, :] = opR(store_residual[:, :, j, :, :]) #this line is not backproping!

        # now need to reshape vlad? Although one could argue that maybe it would be better to keep the region id as a 2-D
        # matrix to make it easier to formulate the region loss function (as region loss is both spatial and visual)

        # this could be a little slow (maybe there is a nicer way?)
        #        for H in range(vlad.shape[3]):
        #            for W in range(vlad.shape[4]):
        #                vlad[:, :, :, H, W] = F.normalize(vlad[:, :, :, H, W], p=2, dim=2)  #or possibly this line

        vlad = vlad.view(x.size(0), -1, Hout, Wout)  # flatten
        vlad = vlad.view(vlad.size(0), vlad.size(1), -1)

        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize every region descriptor

        vlad_full = F.normalize(vlad_full, p=2, dim=2)
        vlad_full = vlad_full.view(x.size(0), -1)
        vlad_full = F.normalize(vlad_full, p=2, dim=1)
        # swapspace - auto send to disk? (ubuntu has a swapspace utility, not sure if useful or not for dealing with region-netvlad)
        if self.get_labels:
            return vlad, vlad_full, labels_regions
        else:
            return vlad, vlad_full
