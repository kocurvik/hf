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

from utils.geometry import rotation_angle, angle, get_pose
from utils.tables import print_results, print_results_summary
from utils.vis import draw_results_focal_auc, draw_results_focal_med, \
    draw_results_focal_cumdist, draw_results_focal_cumdist_all
from utils.voting import focal_voting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-c', '--case', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    parser.add_argument('-is', '--ignore_score', action='store_true', default=False)
    parser.add_argument('--oracle', action='store_true', default=False)
    parser.add_argument('--range', action='store_true', default=False)
    parser.add_argument('--prior', action='store_true', default=False)
    parser.add_argument('--rebuttal', action='store_true', default=False)
    parser.add_argument('--nlo', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def get_camera_dicts(K_file_path):
    K_file = h5py.File(K_file_path)

    d = {}
    for key, v in K_file.items():
        K = np.array(v)
        # d[key.replace('\\', '/')] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K[0, 2]), 'height': int(2 * K[1,2]), 'params': [(K[0, 0] + K[1, 1]) * 0.5, K[0, 2], K[1, 2]]}
        d[key] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K[0, 2]), 'height': int(2 * K[1,2]), 'params': [(K[0, 0] + K[1, 1]) * 0.5, K[0, 2], K[1, 2]]}

    return d

def get_triplets(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    return [line.strip().split(' ') for line in lines]


def get_result_dict(out, info, img1, img2, img3, R_dict, T_dict, K_dict):
    gt_R_12, gt_t_12 = get_pose(img1, img2, R_dict, T_dict)
    gt_R_13, gt_t_13 = get_pose(img1, img3, R_dict, T_dict)
    gt_R_23, gt_t_23 = get_pose(img2, img3, R_dict, T_dict)

    R_12, t_12 = out.poses.pose12.R, out.poses.pose12.t
    R_13, t_13 = out.poses.pose13.R, out.poses.pose13.t
    R_23, t_23 = out.poses.pose23().R, out.poses.pose23().t

    d = {}
    d['R_12_err'] = rotation_angle(R_12.T @ gt_R_12)
    d['R_13_err'] = rotation_angle(R_13.T @ gt_R_13)
    d['R_23_err'] = rotation_angle(R_23.T @ gt_R_23)

    d['t_12_err'] = angle(t_12, gt_t_12)
    d['t_13_err'] = angle(t_13, gt_t_13)
    d['t_23_err'] = angle(t_23, gt_t_23)

    d['P_12_err'] = max(d['R_12_err'], d['t_12_err'])
    d['P_13_err'] = max(d['R_13_err'], d['t_13_err'])
    d['P_23_err'] = max(d['R_23_err'], d['t_23_err'])

    d['P_err_unscaled'] = max(d['P_12_err'], d['P_13_err'])
    d['P_err'] = max([v for k, v in d.items()])

    gt_focal_1 = K_dict[img1]['params'][0]
    gt_focal_2 = K_dict[img2]['params'][0]
    gt_focal_3 = K_dict[img3]['params'][0]

    d['f1_est'] = out.camera1.focal()
    d['f2_est'] = out.camera2.focal()
    d['f3_est'] = out.camera3.focal()

    d['f1_err'] = np.abs(gt_focal_1 - d['f1_est']) / gt_focal_1
    d['f2_err'] = np.abs(gt_focal_2 - d['f2_est']) / gt_focal_2
    d['f3_err'] = np.abs(gt_focal_3 - d['f3_est']) / gt_focal_3

    # mean_gt_focal = (gt_focal_1 + gt_focal_2 + gt_focal_3) / 3
    # d['f_err'] = np.abs(mean_gt_focal - focal) / mean_gt_focal

    info['inliers'] = []
    d['info'] = info
    return d

def get_result_dict_f_only(f1, f2, f3, info, img1, img2, img3, K_dict):
    d = {}
    d['R_12_err'] = 180
    d['R_13_err'] = 180
    d['R_23_err'] = 180

    d['t_12_err'] = 180
    d['t_13_err'] = 180
    d['t_23_err'] = 180

    d['P_12_err'] = max(d['R_12_err'], d['t_12_err'])
    d['P_13_err'] = max(d['R_13_err'], d['t_13_err'])
    d['P_23_err'] = max(d['R_23_err'], d['t_23_err'])

    d['P_err_unscaled'] = max(d['P_12_err'], d['P_13_err'])
    d['P_err'] = max([v for k, v in d.items()])

    gt_focal_1 = K_dict[img1]['params'][0]
    gt_focal_2 = K_dict[img2]['params'][0]
    gt_focal_3 = K_dict[img3]['params'][0]

    d['f1_err'] = np.abs(gt_focal_1 - f1) / gt_focal_1
    d['f2_err'] = np.abs(gt_focal_2 - f2) / gt_focal_2
    d['f3_err'] = np.abs(gt_focal_3 - f3) / gt_focal_3

    # mean_gt_focal = (gt_focal_1 + gt_focal_2 + gt_focal_3) / 3
    # d['f_err'] = np.abs(mean_gt_focal - focal) / mean_gt_focal

    info['inliers'] = []
    if 'inlier_ratio' not in info.keys():
        info['inlier_ratio'] = 0.0
    if 'refinements' not in info.keys():
        info['refinements'] = 1
    d['info'] = info


    return d
def get_result_dict_prior_case3(sols, img1, img2, img3, K_dict):
    d = {}
    d['R_12_err'] = 180
    d['R_13_err'] = 180
    d['R_23_err'] = 180

    d['t_12_err'] = 180
    d['t_13_err'] = 180
    d['t_23_err'] = 180

    d['P_12_err'] = max(d['R_12_err'], d['t_12_err'])
    d['P_13_err'] = max(d['R_13_err'], d['t_13_err'])
    d['P_23_err'] = max(d['R_23_err'], d['t_23_err'])

    d['P_err_unscaled'] = max(d['P_12_err'], d['P_13_err'])
    d['P_err'] = max([v for k, v in d.items()])

    gt_focal_1 = K_dict[img1]['params'][0]
    gt_focal_2 = K_dict[img2]['params'][0]
    gt_focal_3 = K_dict[img3]['params'][0]

    f1_prior = 1.2 * max(K_dict[img1]['width'], K_dict[img1]['height'])
    f2_prior = 1.2 * max(K_dict[img2]['width'], K_dict[img2]['height'])
    f3_prior = 1.2 * max(K_dict[img3]['width'], K_dict[img3]['height'])
    
    if sols.shape[1] == 0:
        f1 = f1_prior
        f2 = f2_prior
        f3 = f3_prior
    else:
        f3s = sols[0, :]
        f1s = sols[1, :]
        f3_dists = np.abs(f3s  - f3_prior) / f3_prior
        f1_dists = np.abs(f1s  - f1_prior) / f1_prior
        dists = np.sqrt(f1_dists * f3_dists)
        best_i = np.argmin(dists)
        f3 = sols[0, best_i]
        f1 = sols[1, best_i]
        f2 = f1

    d['f1_err'] = np.abs(gt_focal_1 - f1) / gt_focal_1
    d['f2_err'] = np.abs(gt_focal_2 - f2) / gt_focal_2
    d['f3_err'] = np.abs(gt_focal_3 - f3) / gt_focal_3

    # mean_gt_focal = (gt_focal_1 + gt_focal_2 + gt_focal_3) / 3
    # d['f_err'] = np.abs(mean_gt_focal - focal) / mean_gt_focal

    info = {'iterations': 0}
    if 'inlier_ratio' not in info.keys():
        info['inlier_ratio'] = 0.0
    if 'refinements' not in info.keys():
        info['refinements'] = 1
    d['info'] = info


    return d


def eval_experiment(x):
    experiment, iterations, img1, img2, img3, triplet, pair12, pair13, R_dict, T_dict, camera_dicts, case = x

    x1 = triplet[:, 0:2]
    x2 = triplet[:, 2:4]
    x3 = triplet[:, 4:6]

    x1_1 = pair12[:, 0:2]
    x2_1 = pair12[:, 2:4]
    x1_2 = pair13[:, 0:2]
    x3_2 = pair13[:, 2:4]

    num_pts = int(experiment[0])
    if iterations is None:
        ransac_dict = {'max_epipolar_error': 3.0, 'progressive_sampling': False,
                       'min_iterations': 100, 'max_iterations': 1000}
    else:
        ransac_dict = {'max_epipolar_error': 3.0, 'progressive_sampling': False,
                       'min_iterations': iterations, 'max_iterations': iterations}

    bundle_dict = {'verbose': False, 'max_iterations': 0 if 'LO(0)' in experiment else 100}
    pp = np.array(camera_dicts[img1]['params'][-2:])

    # ransac_dict['use_hc'] = '4p3vHCf' in experiment
    ransac_dict['use_homography'] = '3vHf' in experiment

    ransac_dict['lo_iterations'] = find_val('LO', experiment, int, default=25)

    if 'Haikkila' in experiment:
        ransac_dict['use_baseline'] = True

    if '+ FR' in experiment:
        # this sets the range to 50-70 deg FOV
        c = 0.89316373181 # (1/tan(35 deg) + 1/tan(25deg)) / 4
        fo = 0.20051164424 # 1 - 1 / (2 * c * tan(25 deg))
        ransac_dict['f1_gt'] = c * max(camera_dicts[img1]['width'], camera_dicts[img1]['height'])
        ransac_dict['f2_gt'] = c * max(camera_dicts[img2]['width'], camera_dicts[img2]['height'])
        ransac_dict['f3_gt'] = c * max(camera_dicts[img3]['width'], camera_dicts[img3]['height'])
        ransac_dict['f_oracle_threshold'] = fo
    else:
        fo = find_val('FO', experiment, float, default=0.0)

        ransac_dict['f_oracle_threshold'] = fo
        if fo:
            # ransac_dict['f1_gt'] = camera_dicts[img1]['params'][0]
            # ransac_dict['f2_gt'] = camera_dicts[img2]['params'][0]
            # ransac_dict['f3_gt'] = camera_dicts[img3]['params'][0]
            ransac_dict['f1_gt'] = 1.2 * max(camera_dicts[img1]['width'], camera_dicts[img1]['height'])
            ransac_dict['f2_gt'] = 1.2 * max(camera_dicts[img2]['width'], camera_dicts[img2]['height'])
            ransac_dict['f3_gt'] = 1.2 * max(camera_dicts[img3]['width'], camera_dicts[img3]['height'])


    ransac_dict['use_degensac'] = 'degensac' in experiment
    ransac_dict['use_onefocal'] = '6p Ef' in experiment

    ransac_dict['problem'] = case

    camera3 = camera_dicts[img3]
    f3 = camera3['params'][0]

    if case == 1:
        if '6p fEf (pairs)' in experiment:
            start = perf_counter()
            out, info = poselib.estimate_shared_focal_relative_pose(x1_1, x2_1, pp, ransac_dict, bundle_dict)
            info['runtime'] = 1000 * (perf_counter() - start)
            focal = out.camera1.focal()
            result_dict = get_result_dict_f_only(focal, focal, focal, info, img1, img2, img3, camera_dicts)
        else:
            start = perf_counter()
            out, info = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, pp, ransac_dict)
            info['runtime'] = 1000 * (perf_counter() - start)
            result_dict = get_result_dict(out, info, img1, img2, img3, R_dict, T_dict, camera_dicts)
    elif case == 2:
        if '6p fEf (pairs)' in experiment:
            start = perf_counter()
            out, info = poselib.estimate_shared_focal_relative_pose(x1_1, x2_1, pp, ransac_dict, bundle_dict)
            info['runtime'] = 1000 * (perf_counter() - start)
            focal = out.camera1.focal()
            result_dict = get_result_dict_f_only(focal, focal, f3, info, img1, img2, img3, camera_dicts)
        elif '6p Ef (pairs)' in experiment:
            start = perf_counter()
            out, info = poselib.estimate_onefocal_relative_pose(x1_2, x3_2, camera3, pp, ransac_dict, bundle_dict)
            info['runtime'] = 1000 * (perf_counter() - start)
            focal = out.camera1.focal()
            result_dict = get_result_dict_f_only(focal, focal, f3, info, img1, img2, img3, camera_dicts)
        else:
            start = perf_counter()
            out, info = poselib.estimate_three_view_case2_relative_pose(x1, x2, x3, camera3, pp, ransac_dict)
            info['runtime'] = 1000 * (perf_counter() - start)
            result_dict = get_result_dict(out, info, img1, img2, img3, R_dict, T_dict, camera_dicts)
    elif case == 3:
        xx1 = x1 - np.array(camera_dicts[img1]['params'][-2:])
        xx2 = x2 - np.array(camera_dicts[img2]['params'][-2:])
        xx3 = x3 - np.array(camera_dicts[img3]['params'][-2:])

        scale = (np.mean(np.linalg.norm(xx1, axis=1)) + np.mean(np.linalg.norm(xx2, axis=1)) + np.mean(np.linalg.norm(xx3, axis=1))) * np.sqrt(2) / 3

        if 'prior' in experiment:
            ransac_dict['min_iterations'] = 1000
            start = perf_counter()
            H31, info = poselib.estimate_homography(xx3 / scale, xx1 / scale, ransac_dict)
            H32, info = poselib.estimate_homography(xx3 / scale, xx2 / scale, ransac_dict)
            sols = poselib.solver_H3f_case3(H31, H32)
            result_dict = get_result_dict_prior_case3(scale * sols, img1, img2, img3, camera_dicts)
            result_dict['info']['runtime'] = 1000 * (perf_counter() - start)
        else:

            start = perf_counter()
            out, info = poselib.estimate_three_view_shared_focal_relative_pose(xx1, xx2, xx3, np.array([0.0, 0.0]), ransac_dict)
            info['runtime'] = 1000 * (perf_counter() - start)
            result_dict = get_result_dict(out, info, img1, img2, img3, R_dict, T_dict, camera_dicts)
    elif case == 4:
        xx1 = x1 - np.array(camera_dicts[img1]['params'][-2:])
        xx2 = x2 - np.array(camera_dicts[img2]['params'][-2:])
        start = perf_counter()
        out, info = poselib.estimate_three_view_case2_relative_pose(xx1, xx2, x3, camera3, np.array([0, 0]), ransac_dict,
                                                                              bundle_dict)
        info['runtime'] = 1000 * (perf_counter() - start)
        result_dict = get_result_dict(out, info, img1, img2, img3, R_dict, T_dict, camera_dicts)

    result_dict['experiment'] = experiment
    result_dict['img1'] = img1
    result_dict['img2'] = img2
    result_dict['img3'] = img3

    # with open(f'results/{experiment}-{img1}-{img2}-{img3}.json', 'w') as f:
    #     json.dump(result_dict, f)

    return result_dict


def get_K(camera_dicts, img1):
    pp = np.array(camera_dicts[img1]['params'][-2:])
    focal = np.array(camera_dicts[img1]['params'][0])
    K = np.diag([focal, focal, 1])
    K[:2, 2] = pp
    return K

def find_val(str, experiment, type=int, default=0):
    if f'{str}(' in experiment:
        identifier_len = 1 + len(str)
        idx = experiment.find(f'{str}(')
        idx_end = experiment[idx+identifier_len:].find(')')
        return type(experiment[idx+ identifier_len :idx + identifier_len + idx_end])
    else:
        return type(default)



def eval(args):
    dataset_path = args.dataset_path
    matches_basename = os.path.basename(args.feature_file)
    basename = os.path.basename(dataset_path)
    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        # iterations_list = [10, 20, 50, 100, 200, 500, 1000]
        # iterations_list = [10, 20, 50, 100]
    else:
        iterations_list = [None]

    # experiments = ['4pH + 4pH + 3vHf + scale', '4pH + 4pH + 3vHf + p3p', '4pH + 4pH + 3vHf + voting (pairs)',
    #                '4pH + 4pH + 3vHf + voting (triplets)', '6pf + p3p', '6pf + p3p + degensac', '6pf (pairs)',
    #                '6pf (pairs) + degensac']

    if args.case == 1:
        experiments = ['4pH + 4pH + 3vHfc1 + p3p',
                       '6p fEf + p3p', '6p fEf + p3p + degensac',
                       '6p fEf (pairs)', '6p fEf (pairs) + degensac',
                       '4pH + 4pH + 3vHfc1 + p3p + Haikkila']
        if args.rebuttal:
            experiments = ['4pH + 4pH + 3vHfc1 + p3p + Haikkila']
    elif args.case == 2:
        experiments = ['4pH + 4pH + 3vHfc2 + p3p', '6p fEf + p3p', '6p fEf + p3p + degensac', '6p Ef + p3p',
                       '6p fEf (pairs)', '6p fEf (pairs) + degensac', '6p Ef (pairs)']

    elif args.case == 3:
        experiments = ['4pH + 4pH + 3vHfc3 + p3p', '6p fEf + p4pf', '6p fEf + p4pf + degensac']

    elif args.case == 4:
        experiments = ['4pH + 4pH + 3vHfc4 + p3p', '6p Ef + p4pf']

    if args.nlo:
        experiments = [f'{x} + LO(0)' for x in experiments]

    if args.oracle:
        experiments = [f'{x} + FO(0.3)' for x in experiments]

    if args.range:
        experiments = [f'{x} + FR' for x in experiments]
    # experiments = ['6pf + p3p', '6pf + p3p + degensac']
    # experiments = ['4pH + 4pH + 3vHf + p3p', '6pf (pairs)', '6pf (pairs) + degensac + LO(0)', '6pf (pairs) + degensac']

    if args.prior:
        if args.case == 3:
            experiments = ['4pH + 4pH + 3vHfc3 + prior']
        elif args.case == 4:
            experiments = ['4pH + 4pH + 3vHfc4 + prior']


    if args.case == 3:
        json_path = os.path.join('results', f'focal_{basename}-{matches_basename}-c3{"-is" if args.ignore_score else ""}.json')
    else:
        json_path = os.path.join('results',
                                 f'focal_{basename}-{matches_basename}{"-is" if args.ignore_score else ""}.json')

    print(f'json_path: {json_path}')

    if args.load:
        with open(json_path, 'r') as f:
            results = json.load(f)
            triplets = get_triplets(os.path.join(dataset_path, f'{args.feature_file}.txt'))

    else:
        C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))
        triplets = get_triplets(os.path.join(dataset_path, f'{args.feature_file}.txt'))

        camera_dicts = get_camera_dicts(os.path.join(dataset_path, 'K.h5'))

        names = camera_dicts.keys()

        R_path = os.path.join(dataset_path, 'R.h5')
        if os.path.exists(R_path):
            R_file = h5py.File(R_path)
            R_dict = {k.replace('\\', '/'): np.array(v) for k, v in R_file.items()}
        else:
            R_dict = {k: np.eye(3) for k in names}

        T_path = os.path.join(dataset_path, 'T.h5')
        if os.path.exists(T_path):
            T_file = h5py.File(T_path)
            T_dict = {k.replace('\\', '/'): np.array(v) for k, v in T_file.items()}
        else:
            T_dict = {k: np.ones(3) for k in names}

        if args.first is not None:
            triplets = triplets[:args.first]

        def gen_data():
            for triplet in triplets:
                img1, img2, img3 = triplet
                label = f"{img1}-{img2}-{img3}"

                pts = np.array(C_file[label])

                if not args.ignore_score:
                    l = np.all(pts[:, 6:] >= 0.5, axis=1)
                    triplet = pts[l, :6]
                else:
                    triplet = pts[:, :6]

                try:
                    label12 = f'{img1}-{img2}'
                    pair12 = np.array(C_file[label12])
                    if not args.ignore_score:
                        pair12 = pair12[pair12[:, -1] > 0.5]

                    label13 = f'{img1}-{img3}'
                    pair13 = np.array(C_file[label13])
                    if not args.ignore_score:
                        pair13 = pair13[pair13[:, -1] > 0.5]
                except Exception:
                    pair12 = pts[:, :4]
                    pair13 = np.column_stack([pts[:, :2], pts[:, 4:6]])


                R_dict_l = {x: R_dict[x] for x in [img1, img2, img3]}
                T_dict_l = {x: T_dict[x] for x in [img1, img2, img3]}
                camera_dicts_l = {x: camera_dicts[x] for x in [img1, img2, img3]}

                for iterations in iterations_list:
                    for experiment in experiments:
                        yield experiment, iterations, img1, img2, img3, triplet, pair12, pair13, R_dict_l, T_dict_l, camera_dicts_l, args.case

        total_length = len(experiments) * len(triplets) * len(iterations_list)
        print(f"Total runs: {total_length} for {len(triplets)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        print("Done")

    if args.append:
        if os.path.exists(json_path):
            print(f"Appending from: {json_path}")
            with open(json_path, 'r') as f:
                prev_results = json.load(f)
        else:
            print("Prev file not found!")
            prev_results = []

        if args.overwrite:
            prev_results = [x for x in prev_results if x['experiment'] not in experiments]

        results.extend(prev_results)

    for experiment in experiments:
        print(50 * '*')
        print(f'Results for: {experiment}:')
        print_results([r for r in results if r['experiment'] == experiment])

    print(50 * '*')
    print(50 * '*')
    print(50 * '*')
    print_results_summary(results, experiments)

    os.makedirs('results', exist_ok=True)

    if not args.load:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

    title = f'Scene: {os.path.basename(dataset_path)} \n'
    title += f'Matches: {matches_basename} ({len(triplets)} samples)\n'


    fig_save_name = f'{os.path.basename(dataset_path)}_{matches_basename}.png'

    if args.graph:
        draw_results_focal_auc(results, experiments, iterations_list, title=title + 'f AUC-0.1',
                               save=f'figs/graph_auc10f_{fig_save_name}')
        plt.show()
        draw_results_focal_med(results, experiments, iterations_list, title=title + 'f median',
                                  save=f'figs/graph_medf_{fig_save_name}')
        plt.show()
    else:
        # draw_results_focal_cumdist(results, experiments, title=title, save=f'figs/cumdistf_{fig_save_name}')
        draw_results_focal_cumdist_all(results, experiments, title=title, save=f'figs/graph_cumdist_{fig_save_name}')
        plt.show()

if __name__ == '__main__':
    args = parse_args()
    eval(args)