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
from tqdm import tqdm

from utils.geometry import rotation_angle, angle, get_pose, skew

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('feature_file')
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


def count_inliers(F, pts1, pts2, threshold):
    """
    Count the number of inliers based on the symmetric epipolar distance and a threshold.

    Parameters:
    F (numpy.ndarray): 3x3 fundamental matrix.
    pts1 (numpy.ndarray): Nx2 array of points in the first image.
    pts2 (numpy.ndarray): Nx2 array of points in the second image.
    threshold (float): Epipolar threshold for considering a point as an inlier.

    Returns:
    int: Number of inliers.
    """
    distances = symmetric_epipolar_distance(F, pts1, pts2)
    inliers = np.sum(distances < threshold**2)
    return inliers


def get_proportion(x):
    img1, img2, pair12, K1, K2 = x

    x1_1 = pair12[:, 0:2]
    x2_1 = pair12[:, 2:4]

    ransac_dict = {'max_reproj_error': 5.0, 'progressive_sampling': False,
                   'min_iterations': 10000, 'max_iterations': 10000}

    H, info = poselib.estimate_homography(x1_1, x2_1, ransac_dict)
    planar_inliers = info['num_inliers']

    # K1[:2, 2] = 0
    # K2[:2, 2] = 0

    HH = np.linalg.inv(K2) @ H @ K1

    poses, _ = poselib.motion_from_homography(HH)

    max_inliers = 0
    for pose in poses:
        R, t = pose.R, pose.t
        F = np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)
        total_inliers = count_inliers(F, x1_1, x2_1, 5.0)
        if total_inliers > max_inliers:
            max_inliers = total_inliers

    result_dict = {}
    result_dict['img1'] = img1
    result_dict['img2'] = img2
    result_dict['max_inliers'] = max_inliers
    result_dict['planar_inliers'] = planar_inliers
    result_dict['proportion'] = planar_inliers / max_inliers

    return result_dict


def get_K(camera_dicts, img1):
    pp = np.array(camera_dicts[img1]['params'][-2:])
    focal = np.array(camera_dicts[img1]['params'][0])
    K = np.diag([focal, focal, 1])
    K[:2, 2] = pp
    return K


def eval(args):
    dataset_path = args.dataset_path
    matches_basename = os.path.basename(args.feature_file)
    basename = os.path.basename(dataset_path)

    C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
    triplets = get_triplets(os.path.join(dataset_path, f'{args.feature_file}.txt'))
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

            yield img1, img2, pair12, K1, K2

    total_length = len(triplets)
    print(f"Total runs: {total_length} for {len(triplets)} samples")

    if args.num_workers == 1:
        results = [get_proportion(x) for x in tqdm(gen_data(), total=total_length)]
    else:
        pool = Pool(args.num_workers)
        results = [x for x in pool.imap(get_proportion, tqdm(gen_data(), total=total_length))]

    print("Done")

    proportions = [x['proportion'] for x in results]
    print("Mean: ", np.mean(proportions))
    print("Median: ", np.median(proportions))


if __name__ == '__main__':
    args = parse_args()
    eval(args)