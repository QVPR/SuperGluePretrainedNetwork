#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
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
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import numpy as np
from tqdm.auto import tqdm as tqdm
import cv2


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    # print(matched_pts1[:, [1, 0]])

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--skip_rows', type=int, default=0,
        help='Skip that many rows from image pair list')
    parser.add_argument(
        '--superpoint_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the suerpoint stuff')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    opt = parser.parse_args()
    print(opt)

    with open(opt.input_pairs, 'r') as f:
        for _ in range(opt.skip_rows):
            f.readline()
        pairs = [l.split() for l in f.readlines()]

    print('Will evaluate %d pairs' % len(pairs))

    superpoint_dir = Path(opt.superpoint_dir)

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))

    for i, pair in enumerate(tqdm(pairs)):
        name0, name1 = pair[:2]
        if name0.endswith(','):
            name0 = name0[:-1]
        if name1.endswith(','):
            name1 = name1[:-1]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        superpointname0, superpointname1 = name0.replace(Path(name0).suffix, '_superpoints.npz'), name1.replace(Path(name1).suffix, '_superpoints.npz')
        matches_path = output_dir / '{}_{}_bf_matches.npz'.format(name0.replace('/', 'SLASH'), name1.replace('/', 'SLASH'))
        superpointpath0, superpointpath1 = superpoint_dir / superpointname0, superpoint_dir / superpointname1

        # Perform the matching.
        superpoint0 = np.load(superpointpath0)
        kpts0 = superpoint0['keypoints']
        kpts0 = [cv2.KeyPoint(x=p[0], y=p[1], _size=1) for p in kpts0]
        desc0 = superpoint0['descriptors'].T

        superpoint1 = np.load(superpointpath1)
        kpts1 = superpoint1['keypoints']
        kpts1 = [cv2.KeyPoint(x=p[0], y=p[1], _size=1) for p in kpts1]
        desc1 = superpoint1['descriptors'].T

        # Match and get rid of outliers
        m_kp1, m_kp2, matches = match_descriptors(kpts0, desc0, kpts1, desc1)
        H, inliers = compute_homography(m_kp1, m_kp2)

        # Write the matches to disk.
        out_matches = {'inliers': inliers, 'homography': H}
        np.savez(str(matches_path), **out_matches)
