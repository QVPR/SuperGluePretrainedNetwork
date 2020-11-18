# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
import numpy as np
from pathlib import Path


from .superpoint import SuperPoint
from .superglue import SuperGlue
from .models import get_backend, get_model, get_resume_ckpt, input_transform, get_pca_encoding, calc_indices
from .attention import findSalient, attn_calc_indices


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config=None, netvlad_opt=None):
        super().__init__()
        if config is None:
            config = {}
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        if config.get('superglue', {})['weights'] in ['indoor', 'outdoor']:
            self.superglue = SuperGlue(config.get('superglue', {}))
        else:
            path = Path(__file__).parent
            path = path / 'weights/superglue_{}.pth'.format(config.get('superglue', {})['weights'])
            self.superglue = torch.load(str(path))

        if netvlad_opt is not None and netvlad_opt.regionnetvlad:
            self.use_regionnetvlad = True
            self.resume_ckpt, self.resume_pca = get_resume_ckpt(netvlad_opt)
            encoder_dim, encoder = get_backend()
            self.model = get_model(encoder, encoder_dim, netvlad_opt, append_pca_layer=self.resume_pca)
            self.device = 'cuda' if torch.cuda.is_available() and not netvlad_opt.force_cpu else 'cpu'
            self.model = self.model.to(self.device)
            self.image_transformer = input_transform()
            self.pool_size = netvlad_opt.num_pcs
            self.regionSize = netvlad_opt.regionSize
            self.moveAmount = netvlad_opt.moveAmount
            if netvlad_opt.useAttentionScoring:
                self.useAttention = True
                self.adj, self.sumsMap, _, _ = attn_calc_indices(netvlad_opt.regionSize, netvlad_opt.moveAmount)
            else:
                self.useAttention = False

            _, _, _, self.indices = calc_indices(netvlad_opt.regionSize, netvlad_opt.moveAmount)
        else:
            self.use_regionnetvlad = False

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        if self.use_regionnetvlad:
            data0 = self.image_transformer(data['frame0'])
            data0 = torch.unsqueeze(data0.to(self.device), 0)
            image_encoding0 = self.model.encoder(data0)
            vlad_regions0, _ = self.model.pool(image_encoding0)
            vlad_regions_pca0 = get_pca_encoding(self.model, vlad_regions0.permute(2, 0, 1).reshape(-1, vlad_regions0.size(1)),
                                                self.resume_ckpt).reshape(vlad_regions0.size(2), vlad_regions0.size(0),
                                                                          self.pool_size).permute(1, 2, 0)

            desc0 = vlad_regions_pca0[0]
            kps0 = self.indices.T
            if self.useAttention:
                scores0 = findSalient(desc0, self.indices, self.adj, self.sumsMap, self.device)
            else:
                scores0 = np.ones((kps0.shape[0]))
            kps0_imagespace = (kps0 + self.regionSize / 2.0) * 16
            # 'keypoints', 'scores', 'descriptors'
            pred['keypoints0'] = [torch.from_numpy(kps0_imagespace.reshape((-1, 2))).to(self.device).float()]
            pred['scores0'] = [scores0]
            pred['descriptors0'] = [desc0]

            data1 = self.image_transformer(data['frame1'])
            data1 = torch.unsqueeze(data1.to(self.device), 0)
            image_encoding1 = self.model.encoder(data1)
            vlad_regions1, _ = self.model.pool(image_encoding1)
            vlad_regions_pca1 = get_pca_encoding(self.model,
                                                 vlad_regions1.permute(2, 0, 1).reshape(-1, vlad_regions1.size(1)),
                                                 self.resume_ckpt).reshape(vlad_regions1.size(2), vlad_regions1.size(0),
                                                                           self.pool_size).permute(1, 2, 0)

            desc1 = vlad_regions_pca1[0]
            kps1 = self.indices.T
            if self.useAttention:
                scores1 = findSalient(desc1, self.indices, self.adj, self.sumsMap, self.device)
            else:
                scores1 = np.ones((kps1.shape[0]))
            # 'keypoints', 'scores', 'descriptors'
            kps1_imagespace = (kps1 + self.regionSize / 2.0) * 16

            pred['keypoints1'] = [torch.from_numpy(kps1_imagespace.reshape((-1, 2))).to(self.device).float()]
            pred['scores1'] = [scores1]
            pred['descriptors1'] = [desc1]
        else:
            # Extract SuperPoint (keypoints, scores, descriptors) if not provided
            if 'keypoints0' not in data or self.use_regionnetvlad:
                # print('extract kp0')
                pred0 = self.superpoint({'image': data['image0']})
                pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
            if 'keypoints1' not in data or self.use_regionnetvlad:
                # print('extract kp1')
                pred1 = self.superpoint({'image': data['image1']})
                pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # print(pred.keys())
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        # print(pred['keypoints0'][0])
        # torch.Size([1, 1, 480, 640]) torch.Size([1, 1, 480, 640])
        # print(data['image0'].shape, data['image1'].shape)
        # (467, 2) (505, 2)
        # print(np.array(pred['keypoints0'][0].cpu().numpy()).shape, np.array(pred['keypoints1'][0].cpu().numpy()).shape)
        # print(pred['keypoints0'].shape, pred['keypoints1'].shape)
        # (497,) (505,)
        # print(pred['scores0'][0].cpu().numpy().shape)
        # print(pred['scores0'].shape, pred['scores1'].shape)
        # (256, 467) (256, 505)
        # print(pred['descriptors0'][0].cpu().numpy().shape, pred['descriptors1'][0].cpu().numpy().shape)
        data = {**data, **pred}
        # print('data', type(data['keypoints0']), type(data['keypoints0'][0]), data['keypoints0'][0].size(), data['keypoints1'][0].size(), data['descriptors0'][0].size(), data['descriptors1'][0].size(), data['scores0'][0].size(), data['scores1'][0].size())

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}
        # print(pred.keys())

        return pred
