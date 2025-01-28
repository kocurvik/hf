import argparse
import json
import os
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
from time import perf_counter

import poselib
import h5py
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

from utils.geometry import rotation_angle, angle, get_pose, skew

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('dataset_path')

    return parser.parse_args()


def get_triplets(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    return [line.strip().split(' ') for line in lines]


def symmetric_epipolar_distance(F, pts1, pts2):
    """
    Compute the symmetric epipolar distance for each pair of points.

    Parameters:
    F (numpy.ndarray): 3x3 fundamental matrix.
    pts1 (numpy.ndarray): Nx2 array of points in the first image.
    pts2 (numpy.ndarray): Nx2 array of points in the second image.

    Returns:
    numpy.ndarray: Array of symmetric epipolar distances for each pair of points.
    """
    # Convert points to homogeneous coordinates
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))  # Nx3
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))  # Nx3

    # Compute epipolar lines in the second image for pts1
    lines2 = np.dot(F, pts1_h.T).T  # Nx3
    # Compute epipolar lines in the first image for pts2
    lines1 = np.dot(F.T, pts2_h.T).T  # Nx3

    # Distance from pts2 to epipolar lines in the second image
    dist2 = np.abs(np.sum(lines2 * pts2_h, axis=1)) / np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)

    # Distance from pts1 to epipolar lines in the first image
    dist1 = np.abs(np.sum(lines1 * pts1_h, axis=1)) / np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)

    # Symmetric epipolar distance
    symmetric_distance = dist1 ** 2 + dist2 ** 2

    return symmetric_distance


def sampson_error(F, pts1, pts2):
    """
    Compute the Sampson error for each pair of points.

    Parameters:
    F (numpy.ndarray): 3x3 fundamental matrix.
    pts1 (numpy.ndarray): Nx2 array of points in the first image.
    pts2 (numpy.ndarray): Nx2 array of points in the second image.

    Returns:
    numpy.ndarray: Array of Sampson errors for each pair of points.
    """
    # Convert points to homogeneous coordinates
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # Compute the Sampson error
    Fx1 = np.dot(F, pts1_h.T)
    Fx2 = np.dot(F.T, pts2_h.T)

    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    error = np.abs(np.sum(pts2_h.T * Fx1, axis=0)) / np.sqrt(denom)

    return error


def get_proportion(x):
    img1, img2, pair12, K1, K2, tr = x

    x1_1 = pair12[:, 0:2]
    x2_1 = pair12[:, 2:4]

    ransac_dict = {'max_reproj_error': tr, 'progressive_sampling': False,
                   'min_iterations': 10000, 'max_iterations': 10000}

    H, info = poselib.estimate_homography(x1_1, x2_1, ransac_dict)
    planar_inliers = info['num_inliers']

    # K1[:2, 2] = 0
    # K2[:2, 2] = 0

    HH = np.linalg.inv(K2) @ H @ K1

    poses, _ = poselib.motion_from_homography(HH)

    max_inliers_sym = 0
    max_inliers_sampson = 0
    for pose in poses:
        R, t = pose.R, pose.t
        F = np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)
        distances = symmetric_epipolar_distance(F, x1_1, x2_1)
        total_inliers_sym = np.sum(distances < tr ** 2)

        sampson_distances = sampson_error(F, x1_1, x2_1)
        total_inliers_sampson = np.sum(sampson_distances < tr)

        if total_inliers_sym > max_inliers_sym:
            max_inliers_sym = total_inliers_sym

        if total_inliers_sampson > max_inliers_sampson:
            max_inliers_sampson = total_inliers_sampson

    result_dict = {}
    result_dict['img1'] = img1
    result_dict['img2'] = img2
    result_dict['max_inliers'] = max_inliers_sym
    result_dict['max_inliers_sampson'] = max_inliers_sampson
    result_dict['planar_inliers'] = planar_inliers
    result_dict['proportion'] = planar_inliers / (max_inliers_sym + 1e-8)
    result_dict['proportion_sampson'] = planar_inliers / (max_inliers_sampson + 1e-8)

    return result_dict


def get_K(camera_dicts, img1):
    pp = np.array(camera_dicts[img1]['params'][-2:])
    focal = np.array(camera_dicts[img1]['params'][0])
    K = np.diag([focal, focal, 1])
    K[:2, 2] = pp
    return K


def eval(dataset_path, num_workers, t):
    C_file = h5py.File(os.path.join(dataset_path, 'triplets-case1-features_superpoint_noresize_2048-LG.h5'))
    triplets = get_triplets(os.path.join(dataset_path, 'triplets-case1-features_superpoint_noresize_2048-LG.txt'))
    K_file = h5py.File(os.path.join(dataset_path, 'K.h5'))

    def gen_data():
        for triplet in triplets:
            img1, img2, img3 = triplet

            # pts = np.array(C_file[label])

            label12 = f'{img1}-{img2}'
            pair12 = np.array(C_file[label12])
            pair12 = pair12[pair12[:, -1] > 0.5]
            K1 = np.array(K_file[img1])
            K2 = np.array(K_file[img2])

            yield img1, img2, pair12, K1, K2, t

    total_length = len(triplets)
    print(f"Total runs: {total_length} for {len(triplets)} samples")

    if num_workers == 1:
        results = [get_proportion(x) for x in tqdm(gen_data(), total=total_length)]
    else:
        pool = Pool(num_workers)
        results = [x for x in pool.imap(get_proportion, tqdm(gen_data(), total=total_length))]

    print("Done")

    proportions = [x['proportion_sampson'] for x in results]
    proportions_sampson = [x['proportion'] for x in results]

    return proportions, proportions_sampson

if __name__ == '__main__':
    args = parse_args()

    dataset_path = args.dataset_path
    subsets = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]

    for subset in subsets:
        tab = PrettyTable(['Threshold', 'Mean Symmetric', 'Median Symmetric', 'Mean Sampson', 'Median Sampson'])
        tab.float_format = '0.2'
        subset_path = os.path.join(dataset_path, subset)
        print("*****************")
        print("Dataset: ", subset)
        print("*****************")
        for t in np.arange(1, 10):
            p, ps = eval(subset_path, args.num_workers, t)
            tab.add_row([t, 100 * np.mean(p), 100 * np.median(p), 100 * np.mean(ps), 100 * np.median(ps)])

        print(tab)


