from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
import poselib
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib import rc

from dataset_utils.data import colors
# from utils.custom_boxplot import custom_dodge_boxplot
from utils.synth import get_random_scene, get_fs


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


large_size = 24
small_size = 20

# plt.rcParams.update({'figure.autolayout': True})
rc('font',**{'family':'serif','serif':['Times New Roman']})
# rc('font',**{'family':'serif'})
rc('text', usetex=True)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def run_methods(x):
    val, f1, f2, f3, xx1, xx2, xx3 = x

    pp = np.array([0.0, 0.0])
    res = []
    bundle_dict = {'max_iterations': 100}
    ransac_dict = {'min_iterations': 100, 'max_iterations': 100, 'max_epipolar_error': 3.0}
    ransac_dict['lo_iterations'] = 25
    ransac_dict['problem'] = 1

    ransac_dict['use_homography'] = True
    ransac_dict['use_p3p'] = True
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '4pH + 4pH + 3vHfc1 + p3p'})

    ransac_dict['use_p3p'] = True
    ransac_dict['use_homography'] = False
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '6p fEf + p3p'})

    ransac_dict['use_p3p'] = True
    ransac_dict['use_homography'] = False
    ransac_dict['use_degensac'] = True
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '6p fEf + p3p + degensac'})

    ransac_dict['use_degensac'] = False
    ransac_dict['use_homography'] = False
    image, info = poselib.estimate_shared_focal_relative_pose(xx1, xx2, pp, ransac_dict, bundle_dict)
    f_est = image.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '6p fEf (pairs)'})

    ransac_dict['use_homography'] = False
    ransac_dict['use_degensac'] = True
    image, info = poselib.estimate_shared_focal_relative_pose(xx1, xx2, pp, ransac_dict, bundle_dict)
    f_est = image.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '6p fEf (pairs) + degensac'})
    return res

def run_methods_case2(x):
    val, f1, f2, f3, xx1, xx2, xx3 = x

    camera3 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1, 'params': [f3, 0, 0]}

    pp = np.array([0.0, 0.0])
    res = []
    bundle_dict = {'max_iterations': 100}
    ransac_dict = {'min_iterations': 100, 'max_iterations': 100, 'max_epipolar_error': 3.0}
    ransac_dict['lo_iterations'] = 25
    ransac_dict['problem'] = 2

    ransac_dict['use_homography'] = True
    image_triplet, info = poselib.estimate_three_view_case2_relative_pose(xx1, xx2, xx3, camera3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '4pH + 4pH + 3vHfc2 + p3p'})

    ransac_dict['use_homography'] = False
    ransac_dict['use_degensac'] = False
    image_triplet, info = poselib.estimate_three_view_case2_relative_pose(xx1, xx2, xx3, camera3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '6p fEf + p3p'})

    ransac_dict['use_homography'] = False
    ransac_dict['use_degensac'] = True
    image_triplet, info = poselib.estimate_three_view_case2_relative_pose(xx1, xx2, xx3, camera3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '6p fEf + p3p + degensac'})

    ransac_dict['use_homography'] = False
    ransac_dict['use_degensac'] = False
    ransac_dict['use_onefocal'] = True
    image_triplet, info = poselib.estimate_three_view_case2_relative_pose(xx1, xx2, xx3, camera3, pp, ransac_dict, bundle_dict)
    f_est = image_triplet.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'], 'Method': '6p Ef + p3p'})

    ransac_dict['use_onefocal'] = False
    ransac_dict['use_degensac'] = False
    image_pair, info  = poselib.estimate_shared_focal_relative_pose(xx1, xx2, pp, ransac_dict, bundle_dict)
    f_est = image_pair.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'],
                'Method': '6p fEf (pairs)'})

    ransac_dict['use_onefocal'] = False
    ransac_dict['use_degensac'] = True
    image_pair, info  = poselib.estimate_shared_focal_relative_pose(xx1, xx2, pp, ransac_dict, bundle_dict)
    f_est = image_pair.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'],
                'Method': '6p fEf (pairs) + degensac'})

    ransac_dict['use_onefocal'] = True
    image_pair, info  = poselib.estimate_onefocal_relative_pose(xx1, xx3, camera3, pp, ransac_dict, bundle_dict)
    f_est = image_pair.camera1.focal()
    res.append({'val': val, 'f_est': f_est, 'f_err': f_err(f_est, f1), 'inliers': info['num_inliers'],
                'Method': '6p Ef (pairs)'})
    return res

def run_method_case3(x):
    val, f1, f2, f3, xx1, xx2, xx3 = x

    pp = np.array([0.0, 0.0])
    res = []
    bundle_dict = {'max_iterations': 0, 'verbose': False}
    ransac_dict = {'min_iterations': 100, 'max_iterations': 100, 'max_epipolar_error': 3.0}
    ransac_dict['lo_iterations'] = 0
    ransac_dict['problem'] = 3

    ransac_dict['use_homography'] = True
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f1_est = image_triplet.camera1.focal()
    f3_est = image_triplet.camera3.focal()
    res.append({'val': val, 'f_est': f1_est, 'f_err': np.sqrt(f_err(f1_est, f1) * f_err(f3_est, f3)), 'inliers': info['num_inliers'], 'Method': '4pH + 4pH + 3vHfc3 + p3p'})

    ransac_dict['use_homography'] = False
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f1_est = image_triplet.camera1.focal()
    f3_est = image_triplet.camera3.focal()
    res.append({'val': val, 'f_est': f1_est, 'f_err': np.sqrt(f_err(f1_est, f1) * f_err(f3_est, f3)), 'inliers': info['num_inliers'], 'Method': '6p fEf + p4pf'})

    ransac_dict['use_homography'] = False
    ransac_dict['use_degensac'] = True
    image_triplet, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, pp, ransac_dict, bundle_dict)
    f1_est = image_triplet.camera1.focal()
    f3_est = image_triplet.camera3.focal()
    res.append({'val': val, 'f_est': f1_est, 'f_err': np.sqrt(f_err(f1_est, f1) * f_err(f3_est, f3)), 'inliers': info['num_inliers'], 'Method': '6p fEf + p4pf + degensac'})

    return res

def run_method_case4(x):
    val, f1, f2, f3, xx1, xx2, xx3 = x

    camera3 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1, 'params': [f3, 0, 0]}

    pp = np.array([0.0, 0.0])
    res = []
    bundle_dict = {'max_iterations': 0}
    ransac_dict = {'min_iterations': 100, 'max_iterations': 100, 'max_epipolar_error': 3.0}
    ransac_dict['lo_iterations'] = 0
    ransac_dict['problem'] = 4

    ransac_dict['use_homography'] = True
    image_triplet, info = poselib.estimate_three_view_case2_relative_pose(xx1, xx2, xx3, camera3, pp, ransac_dict, bundle_dict)
    f1_est = image_triplet.camera1.focal()
    f2_est = image_triplet.camera2.focal()
    res.append({'val': val, 'f_est': f1_est, 'f_err': np.sqrt(f_err(f1_est, f1) * f_err(f2_est, f2)), 'inliers': info['num_inliers'], 'Method': '4pH + 4pH + 3vHfc4 + p3p'})

    ransac_dict['use_homography'] = False
    image_triplet, info = poselib.estimate_three_view_case2_relative_pose(xx1, xx2, xx3, camera3, pp, ransac_dict, bundle_dict)
    f1_est = image_triplet.camera1.focal()
    f2_est = image_triplet.camera2.focal()
    res.append({'val': val, 'f_est': f1_est, 'f_err': np.sqrt(f_err(f1_est, f1) * f_err(f2_est, f2)), 'inliers': info['num_inliers'], 'Method': '6p Ef + p4pf'})

    return res

eval_funcs = {1: run_methods, 2: run_methods_case2, 3: run_method_case3, 4: run_method_case4}


def plane_box_plot(case=1, load=True, repeats=100, legend_visible=True):
    vals = [1.0, 0.95, 0.9, 0.75, 0.5, 0.25, 0.0]

    path = f'saved/threeview_pose_dominant_plane_case{case}.pkl'

    sigma = 1.0

    if load:
        df = pd.read_pickle(path)
    else:
        def gen_data():
            for val in vals:
                for _ in range(repeats):
                    # x1, x2, x3, _ = get_scene(f, f, f3, R12, t12, R13, t13, 300, dominant_plane=val)
                    f1, f2, f3 = get_fs(case)

                    x1, x2, x3, _ = get_random_scene(f1, f2, f3, 200, dominant_plane=val)


                    xx1 = x1 + sigma * np.random.randn(*(x1.shape))
                    xx2 = x2 + sigma * np.random.randn(*(x2.shape))
                    xx3 = x3 + sigma * np.random.randn(*(x3.shape))

                    xx1, xx2, xx3 = shuffle_portion([xx1, xx2, xx3], 0.25)
                    yield val, f1, f2, f3, xx1, xx2, xx3

        total_length = repeats * len(vals)

        p = Pool(8)
        r = [x for x in p.imap(eval_funcs[case], tqdm(gen_data(), total=total_length))]
        # r = [eval_funcs[case](x) for x in tqdm(gen_data(), total=total_length)]

        res = [item for sublist in r for item in sublist]

        df = pd.DataFrame(res, columns=['val', 'f_est', 'f_err', 'Method', 'inliers'])
        df.to_pickle(path)

    order = vals


    # custom_dodge_boxplot(data=df, x='Noise', y='f_est', hue='Method', dodge=True, order=order, width=0.8)
    # sns.boxplot(data=df, x='val', y='f_est', hue='Method', dodge=True, order=order, width=0.8, palette=colors)
    # xlim = (-0.5, len(vals) - 0.5)
    # plt.xlim(xlim)
    # plt.ylim([0, 1400])
    # plt.plot([-0.5, len(vals) - 0.5], [f, f], 'k:')
    # plt.legend(loc='upper left')
    # plt.ylabel('Estimated $f$')
    # plt.xlabel('Portion of points on dominant plane')
    # plt.tick_params(bottom = False)
    # plt.show()


    plt.figure(frameon=False, figsize=(7, 4.5))
    sns.boxplot(data=df, x='val', y='f_err', hue='Method', dodge=True, order=order, width=0.8, palette=colors, flierprops={'marker': '.'})
    xlim = (-0.5, len(vals) - 0.5)
    plt.xlim(xlim)
    # plt.ylim([0.0, 1.0])
    plt.ylim([3e-4, 1.9])
    plt.yscale('log')
    for x in np.arange(0, len(vals)):
        plt.axvline(x + 0.5, color='0.5', linestyle='-', linewidth=0.25, zorder=2.0)

    # plt.legend(loc='upper left')
    plt.legend([], [], frameon=False)
    plt.ylabel('$\\xi_f$', fontsize=large_size)
    # plt.xlabel('$\\frac{n_p}{n}$', fontsize=large_size)
    plt.xlabel('$n_p / n$', fontsize=large_size)
    plt.tick_params(bottom=False)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    # plt.show()

def noise_box_plot(case=1, dominant_portion=0.5, load=True, repeats=100, legend_visible=True):
    vals = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    path = f'saved/threeview_pose_noise_case{case}_{dominant_portion}.pkl'

    scenes = []
    for _ in range(repeats):
        f1, f2, f3 = get_fs(case)
        x1, x2, x3, _ = get_random_scene(f1, f2, f3, 300, dominant_plane=dominant_portion)
        scenes.append((f1, f2, f3, x1, x2, x3))


    if load:
        df = pd.read_pickle(path)
    else:
        def gen_data():
            for val in vals:
                for f1, f2, f3, x1, x2, x3 in scenes:
                    xx1 = x1 + val * np.random.randn(*(x1.shape))
                    xx2 = x2 + val * np.random.randn(*(x2.shape))
                    xx3 = x3 + val * np.random.randn(*(x3.shape))

                    xx1, xx2, xx3 = shuffle_portion([xx1, xx2, xx3], 0.25)
                    yield val, f1, f2, f3, xx1, xx2, xx3

        total_length = repeats * len(vals)

        p = Pool(8)
        r = [x for x in p.imap(eval_funcs[case], tqdm(gen_data(), total=total_length))]
        # r = [eval_funcs[case](x) for x in tqdm(gen_data(), total=total_length)]


        res = [item for sublist in r for item in sublist]

        df = pd.DataFrame(res, columns=['val', 'f_est', 'f_err', 'Method', 'inliers'])
        df.to_pickle(path)

    order = [sigma for sigma in vals]

    # sns.boxplot(data=df, x='val', y='f_est', hue='Method', dodge=True, order=order, width=0.8, palette=colors)
    # xlim = (-0.5, len(vals) - 0.5)
    # plt.xlim(xlim)
    # plt.ylim([1500, 2500])
    # plt.plot([-0.5, len(vals) - 0.5], [f, f], 'k:')
    # plt.legend(loc='upper left')
    # plt.ylabel('Estimated $f$')
    # plt.xlabel('Noise $\\sigma$')
    # plt.tick_params(bottom=False)
    # plt.show()

    plt.figure(frameon=False, figsize=(7, 4.5))
    sns.boxplot(data=df, x='val', y='f_err', hue='Method', dodge=True, order=order, width=0.8, palette=colors, flierprops={'marker': '.'})
    xlim = (-0.5, len(vals) - 0.5)
    plt.xlim(xlim)
    plt.ylim([3e-5, 1.9])
    plt.yscale('log')
    for x in np.arange(0, len(vals)):
        plt.axvline(x + 0.5, color='0.5', linestyle='-', linewidth=0.25, zorder=2.0)

    # plt.legend(loc='upper left')
    plt.legend([], [], frameon=False)
    plt.ylabel('$\\xi_f$', fontsize=large_size)
    plt.xlabel('$\\sigma$', fontsize=large_size)
    plt.tick_params(bottom=False)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    # plt.show()

def f3_box_plot(case=1, dominant_portion=1.05, load=True, repeats=100, legend_visible=True):
    vals = [0.01, 0.02, 0.05, 0.1, 0.2]
    path = f'saved/threeview_pose_f3err_case{case}_{dominant_portion}.pkl'

    scenes = []
    for _ in range(repeats):
        f1, f2, f3 = get_fs(case)
        x1, x2, x3, _ = get_random_scene(f1, f2, f3, 300, dominant_plane=dominant_portion)
        scenes.append((f, f3, x1, x2, x3))

    sigma = 1.0

    if load:
        df = pd.read_pickle(path)
    else:
        def gen_data():
            for val in vals:
                for f, f3, x1, x2, x3 in scenes:
                    xx1 = x1 + sigma * np.random.randn(*(x1.shape))
                    xx2 = x2 + sigma * np.random.randn(*(x2.shape))
                    xx3 = x3 + sigma * np.random.randn(*(x3.shape))

                    f3_wrong = f3 * (1 + np.sign(np.random.rand() - 0.5) * val)

                    xx1, xx2, xx3 = shuffle_portion([xx1, xx2, xx3], 0.25)
                    yield val, f, f, f3_wrong, xx1, xx2, xx3

        total_length = repeats * len(vals)

        p = Pool(8)
        if case == 1:
            r = [x for x in p.imap(run_methods, tqdm(gen_data(), total=total_length))]
            # r = [run_methods(x) for x in tqdm(gen_data(), total=total_length)]
        elif case == 2:
            # r = [x for x in p.imap(run_methods_case2, tqdm(gen_data(), total=total_length))]
            r = [run_methods_case2(x) for x in tqdm(gen_data(), total=total_length)]
        res = [item for sublist in r for item in sublist]

        df = pd.DataFrame(res, columns=['val', 'f_est', 'f_err', 'Method', 'inliers'])
        df.to_pickle(path)

    order = [sigma for sigma in vals]

    # sns.boxplot(data=df, x='val', y='f_est', hue='Method', dodge=True, order=order, width=0.8, palette=colors)
    # xlim = (-0.5, len(vals) - 0.5)
    # plt.xlim(xlim)
    # plt.ylim([1500, 2500])
    # plt.plot([-0.5, len(vals) - 0.5], [f, f], 'k:')
    # plt.legend(loc='upper left')
    # plt.ylabel('Estimated $f$')
    # plt.xlabel('Noise $\\sigma$')
    # plt.tick_params(bottom=False)
    # plt.show()

    plt.figure(frameon=False, figsize=(7, 4.5))
    sns.boxplot(data=df, x='val', y='f_err', hue='Method', dodge=True, order=order, width=0.8, palette=colors, flierprops={'marker': '.'})
    xlim = (-0.5, len(vals) - 0.5)
    plt.xlim(xlim)
    plt.ylim([3e-4, 1.9])
    plt.yscale('log')
    for x in np.arange(0, len(vals)):
        plt.axvline(x + 0.5, color='0.5', linestyle='-', linewidth=0.25, zorder=2.0)

    # plt.legend(loc='upper left')
    plt.legend([], [], frameon=False)
    plt.ylabel('$\\xi_f$', fontsize=large_size)
    plt.xlabel('$\\xi_\\rho$', fontsize=large_size)

    plt.tick_params(bottom=False)
    plt.tick_params(axis='x', which='major', labelsize=small_size)
    plt.tick_params(axis='y', which='major', labelsize=small_size)
    # plt.show()

if __name__ == '__main__':
    plane_box_plot(case=1, repeats=100, load=False)
    plt.savefig('figs/synth/case1_plane.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=1, dominant_portion=1.0, repeats=100, load=False)
    plt.savefig('figs/synth/case1_noise_plane10.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=1, dominant_portion=0.5, repeats=100, load=False)
    plt.savefig('figs/synth/case1_noise_plane05.pdf', pad_inches=0, bbox_inches='tight')


    plane_box_plot(case=2, repeats=100, load=True)
    plt.savefig('figs/synth/case2_plane.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=2, dominant_portion=1.0, repeats=100, load=True)
    plt.savefig('figs/synth/case2_noise_plane10.pdf', pad_inches=0, bbox_inches='tight')

    f3_box_plot(case=2, dominant_portion=1.0, repeats=100, load=True)
    plt.savefig('figs/synth/case2_f3err.pdf', pad_inches=0, bbox_inches='tight')

    plane_box_plot(case=3, repeats=100, load=False)
    plt.savefig('figs/synth/case3_plane.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=3, dominant_portion=1.0, repeats=100, load=False)
    plt.savefig('figs/synth/case3_noise_plane10.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=3, dominant_portion=0.95, repeats=100, load=False)
    plt.savefig('figs/synth/case3_noise_plane95.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=3, dominant_portion=0.5, repeats=100, load=False)
    plt.savefig('figs/synth/case3_noise_plane05.pdf', pad_inches=0, bbox_inches='tight')

    plane_box_plot(case=4, repeats=100, load=False)
    plt.savefig('figs/synth/case4_plane.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=4, dominant_portion=1.0, repeats=100, load=False)
    plt.savefig('figs/synth/case4_noise_plane10.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=4, dominant_portion=0.95, repeats=100, load=False)
    plt.savefig('figs/synth/case4_noise_plane95.pdf', pad_inches=0, bbox_inches='tight')

    noise_box_plot(case=4, dominant_portion=0.5, repeats=100, load=False)
    plt.savefig('figs/synth/case4_noise_plane05.pdf', pad_inches=0, bbox_inches='tight')


