from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
import poselib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# from utils.custom_boxplot import custom_dodge_boxplot
from utils.synth import get_scene


def noise_box_plot(load=True, ylim=(0, 1400), repeats=20, legend_visible=True):
    noise_vals = [0.0, 1.0, 2.0, 5.0, 10.0]

    path = 'saved/threeview_pose_noise.pkl'

    # thetas = np.random.rand(repeats) * 30 - 15
    # ys = np.random.rand(repeats) * 400 - 200
    # thetas = np.random.randn(repeats) * 10.0
    # ys = np.random.randn(repeats) * 200

    f = 600
    R12 = Rotation.from_euler('xyz', (3, 60, 0), degrees=True).as_matrix()
    R13 = Rotation.from_euler('xyz', (-5, -30, 0), degrees=True).as_matrix()
    c1 = np.array([2 * f, 0, f])
    c2 = np.array([0, f, 0.5 * f])
    t12 = -R12 @ c1
    t13 = -R13 @ c2

    scenes = [get_scene(f, R12, t12, R13, t13, 300, dominant_plane=0.5) for _ in range(repeats)]

    pp = np.array([0.0, 0.0])

    res = []

    if load:
        df = pd.read_pickle(path)
    else:
        for sigma in tqdm(noise_vals):
            for x1, x2, x3, _ in tqdm(scenes):
                xx1 = x1 + sigma * np.random.randn(*(x1.shape))
                xx2 = x2 + sigma * np.random.randn(*(x2.shape))
                xx3 = x3 + sigma * np.random.randn(*(x3.shape))

                ransac_dict = {'min_iterations': 1000, 'max_iterations': 1000, 'max_epipolar_error': 1.0}

                ransac_dict['relpose_scale'] = True
                ransac_dict['use_homography'] = False
                image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict)
                f_est = image_triplet.camera.focal()
                res.append({'Noise': sigma, 'f_est': f_est, 'Method': '6pf + p3p'})

                ransac_dict['relpose_scale'] = True
                ransac_dict['use_homography'] = True
                image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict)
                f_est = image_triplet.camera.focal()
                res.append({'Noise': sigma, 'f_est': f_est, 'Method': '4pH + 4pH + 3vHf'})

        df = pd.DataFrame(res, columns=['Noise', 'f_est', 'Method'])
        df.to_pickle(path)

    order = [sigma for sigma in noise_vals]


    # custom_dodge_boxplot(data=df, x='Noise', y='f_est', hue='Method', dodge=True, order=order, width=0.8)
    sns.boxplot(data=df, x='Noise', y='f_est', hue='Method', dodge=True, order=order, width=0.8)

    axes = plt.gca()
    xlim = (-0.5, len(noise_vals) - 0.5)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='upper left')
    plt.plot(xlim, [f, f], 'k-')

    plt.ylabel('Estimated $f$')
    plt.xlabel('$\sigma_n$')
    plt.tick_params(bottom = False)


def shuffle_portion(x_list, s):
    num_rows_to_shuffle = int(s * x_list[0].shape[0])
    indices_to_shuffle = np.random.choice(x_list[0].shape[0], num_rows_to_shuffle, replace=False)

    new_list = []

    for x in x_list:

        rows_to_shuffle = x[indices_to_shuffle]
        np.random.shuffle(rows_to_shuffle)
        shuffled_x = x.copy()
        shuffled_x[indices_to_shuffle] = rows_to_shuffle
        new_list.append(shuffled_x)

    return new_list


def f_err(f_est, f_gt):
    return np.abs(f_est-f_gt) / f_gt


def run_methods(x):

    val, f, xx1, xx2, xx3 = x

    pp = np.array([0.0, 0.0])
    res = []
    bundle_dict = {'max_iterations': 100}
    ransac_dict = {'min_iterations': 1000, 'max_iterations': 1000, 'max_epipolar_error': 2.0}
    ransac_dict['lo_iterations'] = 25

    ransac_dict['relpose_scale'] = True
    ransac_dict['use_homography'] = True
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f), 'inliers': info['num_inliers'], 'Method': '4pH + 4pH + 3vHf'})

    ransac_dict['relpose_scale'] = True
    ransac_dict['use_homography'] = False
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f), 'inliers': info['num_inliers'], 'Method': '6pf + p3p'})

    ransac_dict['relpose_scale'] = False
    ransac_dict['use_homography'] = False
    image, info = poselib.estimate_shared_focal_relative_pose(xx1, xx2, pp, ransac_dict, bundle_dict)
    f_est = image.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f), 'inliers': info['num_inliers'], 'Method': '6pf'})
    return res

def plane_box_plot(load=True, repeats=100, legend_visible=True):
    vals = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.5, 0.25, 0.0]

    path = 'saved/threeview_pose_dominant_plane.pkl'

    # thetas = np.random.rand(repeats) * 30 - 15
    # ys = np.random.rand(repeats) * 400 - 200
    # thetas = np.random.randn(repeats) * 10.0
    # ys = np.random.randn(repeats) * 200

    f = 600
    R12 = Rotation.from_euler('xyz', (-3, 60, 0), degrees=True).as_matrix()
    R13 = Rotation.from_euler('xyz', (5, -30, 0), degrees=True).as_matrix()
    c1 = np.array([2 * f, 0, f])
    c2 = np.array([0, f, 0.5 * f])
    t12 = -R12 @ c1
    t13 = -R13 @ c2

    sigma = 1.0

    if load:
        df = pd.read_pickle(path)
    else:

        def gen_data():
            for val in vals:
                for _ in range(repeats):
                    x1, x2, x3, _ = get_scene(f, R12, t12, R13, t13, 300, dominant_plane=val)

                    xx1 = x1 + sigma * np.random.randn(*(x1.shape))
                    xx2 = x2 + sigma * np.random.randn(*(x2.shape))
                    xx3 = x3 + sigma * np.random.randn(*(x3.shape))

                    xx1, xx2, xx3 = shuffle_portion([xx1, xx2, xx3], 0.1)
                    yield val, f, xx1, xx2, xx3

        total_length = repeats * len(vals)

        p = Pool(8)
        r = [x for x in p.imap(run_methods, tqdm(gen_data(), total=total_length))]
        res = [item for sublist in r for item in sublist]

        df = pd.DataFrame(res, columns=['val', 'f_est', 'f_err', 'Method', 'inliers'])
        df.to_pickle(path)

    order = vals


    # custom_dodge_boxplot(data=df, x='Noise', y='f_est', hue='Method', dodge=True, order=order, width=0.8)
    sns.boxplot(data=df, x='val', y='f_est', hue='Method', dodge=True, order=order, width=0.8)
    xlim = (-0.5, len(vals) - 0.5)
    plt.xlim(xlim)
    plt.ylim([0, 1400])
    plt.plot([-0.5, len(vals) - 0.5], [f, f], 'k:')
    plt.legend(loc='upper left')
    plt.ylabel('Estimated $f$')
    plt.xlabel('Portion of points on dominant plane')
    plt.tick_params(bottom = False)
    plt.show()

    sns.boxplot(data=df, x='val', y='f_err', hue='Method', dodge=True, order=order, width=0.8)
    xlim = (-0.5, len(vals) - 0.5)
    plt.xlim(xlim)
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.ylabel('$\\frac{|f_{est} - f_{GT}|}{f_{GT}}$')
    plt.xlabel('Portion of points on dominant plane')
    plt.tick_params(bottom=False)
    plt.show()

    sns.boxplot(data=df, x='val', y='inliers', hue='Method', dodge=True, order=order, width=0.8)
    xlim = (-0.5, len(vals) - 0.5)
    plt.xlim(xlim)
    plt.ylim([0, 300])
    plt.legend(loc='upper left')
    plt.ylabel('$\\frac{|f_{est} - f_{GT}|}{f_{GT}}$')
    plt.xlabel('Portion of points on dominant plane')
    plt.tick_params(bottom=False)
    plt.show()

if __name__ == '__main__':
    plane_box_plot(load=False)
    plt.show()
