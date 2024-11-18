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
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
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


def get_results_sample(case, out, img1, img2, img3, K_dict):
    gt_focal_1 = K_dict[img1]['params'][0]
    gt_focal_2 = K_dict[img2]['params'][0]
    gt_focal_3 = K_dict[img3]['params'][0]

    errs = []
    for image_triplets in out:
        if case == 3:
            f_errs = [np.sqrt(abs(it.camera3.focal() - gt_focal_3)/gt_focal_3 * abs(it.camera1.focal() - gt_focal_1)/gt_focal_1) for it in image_triplets]
        elif case == 4:
            f_errs = [np.sqrt(abs(it.camera1.focal() - gt_focal_1) / gt_focal_1 * abs(it.camera2.focal() - gt_focal_2) / gt_focal_2) for it in image_triplets]
        # f_errs = [abs(it.camera2.focal() - gt_focal_2)/gt_focal_2 for it in image_triplets]

        if len(f_errs) > 0:
            errs.append(min(f_errs))

    return errs

def eval_experiment(x):
    experiment, img1, img2, img3, triplet, R_dict, T_dict, camera_dicts, case = x

    num_pts = int(experiment[0])
    ransac_dict = {'min_iterations': 100,
                   'max_iterations': 100 }

    pp1 = np.array(camera_dicts[img1]['params'][-2:])
    pp2 = np.array(camera_dicts[img2]['params'][-2:])
    pp3 = np.array(camera_dicts[img3]['params'][-2:])

    x1 = triplet[:, 0:2] - pp1
    x2 = triplet[:, 2:4] - pp2
    x3 = triplet[:, 4:6] - pp3

    # ransac_dict['use_hc'] = '4p3vHCf' in experiment
    ransac_dict['use_homography'] = '3vHf' in experiment

    # ransac_dict['f1_gt'] = camera_dicts[img1]['params'][0]
    # ransac_dict['f2_gt'] = camera_dicts[img2]['params'][0]
    # ransac_dict['f3_gt'] = camera_dicts[img3]['params'][0]

    ransac_dict['use_degensac'] = 'degensac' in experiment
    ransac_dict['use_onefocal'] = '6p Ef' in experiment

    ransac_dict['problem'] = case

    camera3 = camera_dicts[img3].copy()
    camera3['params'][-2] = 0.0
    camera3['params'][-1] = 0.0

    if case == 3:
        out = poselib.sample_threeview_focal_case3(x1, x2, x3, ransac_dict)
    elif case == 4:
        out = poselib.sample_threeview_focal_case4(x1, x2, x3, camera3, ransac_dict)

    f_errs = get_results_sample(case, out, img1, img2, img3, camera_dicts)

    result_dict = {'experiment': experiment, 'img1': img1, 'img2': img2, 'img3': img3, 'f_errs': f_errs}

    # with open(f'results/{experiment}-{img1}-{img2}-{img3}.json', 'w') as f:
    #     json.dump(result_dict, f)

    return result_dict


def draw_sample_histogram(results, experiments):
    plt.figure()
    for exp in experiments:
        exp_results = [x['f_errs'] for x in results if x['experiment'] == exp]

        f_errs = [x for xs in exp_results for x in xs]

        bins = np.linspace(0.0, 1.5, 100)
        vals, bin_edges = np.histogram(f_errs, bins=20, range=(0, 1.0), density=True)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.plot(x, vals, label=exp)
        # plt.xscale('log')

    plt.xlim((0, 1))
    plt.ylim((0, 2))

    plt.legend()
    plt.show()


def eval(args):
    dataset_path = args.dataset_path
    matches_basename = os.path.basename(args.feature_file)
    basename = os.path.basename(dataset_path)

    if args.case == 3:
        experiments = ['4pH + 4pH + 3vHfc3 + p3p', '6p fEf + p4pf', '6p fEf + p4pf + degensac']

    elif args.case == 4:
        experiments = ['4pH + 4pH + 3vHfc4 + p3p', '6p Ef + p4pf']

    if args.case == 3:
        json_path = os.path.join('sample_results', f'focal_{basename}-{matches_basename}-c3.json')
    else:
        json_path = os.path.join('sample_results', f'focal_{basename}-{matches_basename}.json')

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
                triplet = pts[:, :6]

                R_dict_l = {x: R_dict[x] for x in [img1, img2, img3]}
                T_dict_l = {x: T_dict[x] for x in [img1, img2, img3]}
                camera_dicts_l = {x: camera_dicts[x] for x in [img1, img2, img3]}

                for experiment in experiments:
                    yield experiment, img1, img2, img3, triplet, R_dict_l, T_dict_l, camera_dicts_l, args.case

        total_length = len(experiments) * len(triplets)
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

    # for experiment in experiments:
    #     print(50 * '*')
    #     print(f'Results for: {experiment}:')
    #     print_results([r for r in results if r['experiment'] == experiment])

    # print(50 * '*')
    # print(50 * '*')
    # print(50 * '*')
    # print_results_summary(results, experiments)

    os.makedirs('results', exist_ok=True)

    if not args.load:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

    title = f'Scene: {os.path.basename(dataset_path)} \n'
    title += f'Matches: {matches_basename} ({len(triplets)} samples)\n'


    fig_save_name = f'{os.path.basename(dataset_path)}_{matches_basename}.png'

    draw_sample_histogram(results, experiments)



if __name__ == '__main__':
    args = parse_args()
    eval(args)