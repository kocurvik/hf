import argparse
import itertools
import json
import ntpath
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm
import lightglue
from lightglue.utils import load_image, numpy_image_to_torch

from dataset_utils.data import is_image
from utils.images import load_rotated_image
from utils.matching import LoFTRMatcher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=None)
    parser.add_argument('-s', '--seed', type=int, default=100)
    parser.add_argument('-c', '--case', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--features', type=str, default='superpoint')
    parser.add_argument('-mf', '--max_features', type=int, default=2048)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-a', '--append', action='store_true', default=False)
    parser.add_argument('--scene', type=str, default=None)

    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()

def create_gt_h5(img_dict, calib_dict, out_dir, args):
    if os.path.exists(os.path.join(out_dir, 'K.h5'))  and not args.recalc:
        print(f"GT info exists in {out_dir} - not creating it anew")
        return

    print(f"Writing GT info to {out_dir}")
    fK = h5py.File(os.path.join(out_dir, 'K.h5'), 'w')

    for camera, img_list in img_dict.items():
        K = np.array(calib_dict[camera]['K'])
        K[0, 2] = calib_dict[camera]['width'] / 2
        K[1, 2] = calib_dict[camera]['height'] / 2
        for img in img_list:
            fK.create_dataset(ntpath.normpath(img), shape=(3, 3), data=K)

def get_matcher_string(args):
    if args.resize is None:
        resize_str = 'noresize'
    else:
        resize_str = str(args.resize)

    return f'features_{args.features}_{resize_str}_{args.max_features}'

def get_extractor(args):
    if args.features == 'superpoint':
        extractor = lightglue.SuperPoint(max_num_keypoints=args.max_features).eval().cuda()
    elif args.features == 'disk':
        extractor = lightglue.DISK(max_num_keypoints=args.max_features).eval().cuda()
    elif args.features == 'sift':
        extractor = lightglue.SIFT(max_num_keypoints=args.max_features).eval().cuda()
    else:
        raise NotImplementedError

    return extractor

def extract_features(dataset_path, img_dict, calib_dict, out_dir, args):
    extractor = get_extractor(args)
    out_path = os.path.join(out_dir, f"{get_matcher_string(args)}.pt")


    if os.path.exists(out_path) and not args.recalc:
        print(f"Features already found in {out_path}")
        return

    print("Extracting features")

    if args.append:
        feature_dict = torch.load(out_path)
        print("Loaded features from: ", out_path)
    else:
        feature_dict = {}

    for camera, img_list in img_dict.items():
        width = calib_dict[camera]['width']
        height = calib_dict[camera]['height']
        for img in tqdm(img_list):
            if args.append:
                if f"{ntpath.normpath(img)}-0" in feature_dict.keys() \
                        and f"{ntpath.normpath(img)}-1" in feature_dict.keys() \
                        and f"{ntpath.normpath(img)}-2" in feature_dict.keys() \
                        and f"{ntpath.normpath(img)}-3" in feature_dict.keys():
                    continue

            img_path = os.path.join(dataset_path, img)

            image_tensor = numpy_image_to_torch(load_rotated_image(img_path)[:, :, ::-1]).cuda()
            # image_tensor = load_image(img_path).cuda()

            if image_tensor.size(1) != height or image_tensor.size(2) != width:
                raise ValueError(f"Image {image_tensor.size(2)} x {image_tensor.size(1)}, but expected {width} x {height} "
                                 f"for {img_path}")

            for i in range(4):
                kp_tensor = extractor.extract(torch.rot90(image_tensor, i, dims=[1, 2]), resize=args.resize)
                feature_dict[f"{ntpath.normpath(img)}-{i}"] = kp_tensor

    torch.save(feature_dict, out_path)
    print("Features saved to: ", out_path)


def rot_f(width, height, k):
    k = k % 4

    if k == 0:
        def f(p):
            return np.array([p[0], p[1]])

    elif k == 1:
        def f(p):
            return np.array([width - p[1] - 1, p[0]])

    elif k == 2:
        def f(p):
            return np.array([width - p[0] - 1, height - p[1] - 1])

    elif k == 3:
        def f(p):
            return np.array([p[1], height - p[0] - 1])

    return f


def create_triplets(out_dir, img_dict, calib_dict, args, img_list=None):
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = 0

    features = torch.load(os.path.join(out_dir, f"{get_matcher_string(args)}.pt"))

    matcher = lightglue.LightGlue(features=args.features).eval().cuda()

    triplet_h5_path_str = f'triplets-case{args.case}-{get_matcher_string(args)}-LG.h5'
    triplet_h5_path = os.path.join(out_dir, triplet_h5_path_str)
    triplet_h5_file = h5py.File(triplet_h5_path, 'w')
    print("Writing triplet matches to: ", triplet_h5_path)

    triplets = []

    sample_list = []

    if args.case == 1:
        for camera, image_list in img_dict.items():
            sample_list.append([camera, image_list, camera, image_list, camera, image_list])
    elif args.case == 2:
        for camera12, camera3 in itertools.combinations(list(img_dict.keys()), 2):
            sample_list.append([camera12, img_dict[camera12], camera12, img_dict[camera12], camera3, img_dict[camera3]])
    # case 3 is the same as case 2
    elif args.case == 4:
        for camera1, camera2, camera3 in itertools.combinations(list(img_dict.keys()), 3):
            sample_list.append([camera1, img_dict[camera1], camera2, img_dict[camera2], camera3, img_dict[camera3]])

    with tqdm(total=args.num_samples * len(sample_list)) as pbar:
        for camera1, list1, camera2, list2, camera3, list3 in sample_list:
            width1, height1 = calib_dict[camera1]['width'], calib_dict[camera1]['height']
            width2, height2 = calib_dict[camera2]['width'], calib_dict[camera2]['height']
            width3, height3 = calib_dict[camera3]['width'], calib_dict[camera3]['height']
            output = 0
            failures = 0
            while output < args.num_samples:
                if failures > 100:
                    print(f"Failed to find enough triplets for camera: {camera1}-{camera2}-{camera3}, moving on with {output} triplets")
                    break

                img_triplet = [random.choice(list1), random.choice(list2), random.choice(list3)]
                if len(set(img_triplet)) < 3:
                    failures +=1
                    continue

                img_triplet = [ntpath.normpath(x) for x in img_triplet]
                triplet_label = '-'.join(img_triplet)

                if triplet_label in triplet_h5_file:
                    failures += 1
                    continue

                img_1, img_2, img_3 = img_triplet

                feats_1 = features[f"{img_1}-0"]

                feats_2 = [None] * 4
                scores_12 = [None] * 4
                matches_12 = [None] * 4
                for k in range(4):
                    feats_2[k] = features[f"{img_2}-{k}"]
                    out_12 = matcher({'image0': feats_1, 'image1': feats_2[k]})
                    matches_12[k] = out_12['matches0'][0].detach().cpu().numpy()
                    scores_12[k] = out_12['matching_scores0'][0].detach().cpu().numpy()
                best_k_2 = np.argmax([np.sum(s > 0.5) for s in scores_12])

                feats_2 = feats_2[best_k_2]
                matches_12 = matches_12[best_k_2]
                scores_12 = scores_12[best_k_2]
                ft2 = rot_f(width2, height2, best_k_2)

                feats_3 = [None] * 4
                scores_13 = [None] * 4
                matches_13 = [None] * 4
                for k in range(4):
                    feats_3[k] = features[f"{img_3}-{k}"]
                    out_13 = matcher({'image0': feats_1, 'image1': feats_3[k]})
                    matches_13[k] = out_13['matches0'][0].detach().cpu().numpy()
                    scores_13[k] = out_13['matching_scores0'][0].detach().cpu().numpy()
                best_k_3 = np.argmax([np.sum(s > 0.5) for s in scores_13])

                feats_3 = feats_3[best_k_3]
                matches_13 = matches_13[best_k_3]
                scores_13 = scores_13[best_k_3]
                ft3 = rot_f(width3, height3, best_k_3)

                out_23 = matcher({'image0': feats_2, 'image1': feats_3})
                scores_23 = out_23['matching_scores0'][0].detach().cpu().numpy()
                matches_23 = out_23['matches0'][0].detach().cpu().numpy()

                # TRIPLETS

                idxs = []

                for idx_1, idx_2 in enumerate(matches_12):
                    if idx_2 == -1:
                        continue
                    if matches_13[idx_1] == -1:
                        continue
                    idx_3 = matches_13[idx_1]

                    if matches_23[idx_2] != idx_3:
                        continue

                    idxs.append((idx_1, idx_2, idx_3))

                out_triplet_array = np.empty([len(idxs), 9])

                for i, x in enumerate(idxs):
                    idx_1, idx_2, idx_3 = x
                    point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
                    point_2 = ft2(feats_2['keypoints'][0, idx_2].detach().cpu().numpy())
                    point_3 = ft3(feats_3['keypoints'][0, idx_3].detach().cpu().numpy())
                    score = scores_12[idx_1]
                    score_13 = scores_13[idx_1]
                    score_23 = scores_23[idx_2]
                    out_triplet_array[i] = np.array([*point_1, *point_2, *point_3, score, score_13, score_23])

                if args.debug:
                    kp1 = out_triplet_array[:, :2]
                    kp2 = out_triplet_array[:, 2:4]
                    kp3 = out_triplet_array[:, 4:6]
                    img1 = load_rotated_image(os.path.join(args.dataset_path, img_1.replace(ntpath.sep, os.path.sep)))
                    img2 = load_rotated_image(os.path.join(args.dataset_path, img_2.replace(ntpath.sep, os.path.sep)))
                    img3 = load_rotated_image(os.path.join(args.dataset_path, img_3.replace(ntpath.sep, os.path.sep)))
                    joint_img = np.zeros([np.max([height1, height2, height3]), width1 + width2 + width3, 3], dtype=np.uint8)
                    joint_img[:height1, :width1, :] = img1
                    joint_img[:height2, width1:width1 + width2, :] = img2
                    joint_img[:height3, width1+width2:width1+width2+width3, :] = img3
                    joint_img = np.concatenate([joint_img, joint_img], axis=0)
                    for i in range(len(kp1)):
                        joint_img = cv2.line(joint_img, (int(kp1[i][0]), int(kp1[i][1])), (int(kp2[i][0]) + img1.shape[1], int(kp2[i][1])), (0, 255, 0), thickness=3)
                        joint_img = cv2.line(joint_img, (int(kp2[i][0]) + img1.shape[1], int(kp2[i][1])), (int(kp3[i][0]) + img1.shape[1] + img2.shape[1], int(kp3[i][1])), (0, 255, 0), thickness=3)
                    cv2.namedWindow("Joint img", cv2.WINDOW_NORMAL)
                    cv2.imshow("Joint img", joint_img)
                    cv2.waitKey(0)


                if len(idxs) < 10:
                    failures += 1
                    continue

                triplet_h5_file.create_dataset(triplet_label, shape=out_triplet_array.shape, data=out_triplet_array)
                triplets.append(triplet_label.replace('-', ' '))

                # add pairs if not present already

                pair_label = f'{img_1}-{img_2}'
                if not pair_label in triplet_h5_file:
                    out_array = []
                    for idx_1, idx_2 in enumerate(matches_12):
                        if idx_2 == -1:
                            continue
                        point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
                        point_2 = ft2(feats_2['keypoints'][0, idx_2].detach().cpu().numpy())
                        score = scores_12[idx_1]
                        out_array.append(np.array([*point_1, *point_2, score]))
                    out_array = np.array(out_array)
                    triplet_h5_file.create_dataset(pair_label, shape=out_array.shape, data=out_array)

                pair_label = f'{img_1}-{img_3}'
                if not pair_label in triplet_h5_file:
                    out_array = []
                    for idx_1, idx_2 in enumerate(matches_13):
                        if idx_2 == -1:
                            continue
                        point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
                        point_2 = ft3(feats_3['keypoints'][0, idx_2].detach().cpu().numpy())
                        score = scores_13[idx_1]
                        out_array.append(np.array([*point_1, *point_2, score]))
                    out_array = np.array(out_array)
                    triplet_h5_file.create_dataset(pair_label, shape=out_array.shape, data=out_array)

                pair_label = f'{img_2}-{img_3}'
                if not pair_label in triplet_h5_file:
                    out_array = []
                    for idx_1, idx_2 in enumerate(matches_23):
                        if idx_2 == -1:
                            continue
                        point_1 = ft2(feats_2['keypoints'][0, idx_1].detach().cpu().numpy())
                        point_2 = ft3(feats_3['keypoints'][0, idx_2].detach().cpu().numpy())
                        score = scores_23[idx_1]
                        out_array.append(np.array([*point_1, *point_2, score]))
                    out_array = np.array(out_array)
                    triplet_h5_file.create_dataset(pair_label, shape=out_array.shape, data=out_array)

                if args.num_samples is not None:
                    pbar.update(1)
                    output += 1
                    failures = 0

    triples_txt_path = os.path.join(out_dir, f'triplets-case{args.case}-{get_matcher_string(args)}-LG.txt')
    print("Writing list of triplets to: ", triples_txt_path)
    with open(triples_txt_path, 'w') as f:
        f.writelines(line + '\n' for line in triplets)


def prepare_single(args, scene, camera_list, calib_dict):
    out_dir = os.path.join(args.out_path, scene)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_dict = {}
    for camera in camera_list:
        subset_path = os.path.join(args.dataset_path, camera, scene)
        img_dict[camera] = [os.path.join(camera, scene, x) for x in os.listdir(subset_path) if is_image(x)]

    create_gt_h5(img_dict, calib_dict, out_dir, args)
    # if 'loftr' in args.features:
    #     create_triplets_loftr(out_dir, img_dict, calib_dict, args)
    # else:
    extract_features(args.dataset_path, img_dict, calib_dict, out_dir, args)
    create_triplets(out_dir, img_dict, calib_dict, args)

def run_im(args):
    dataset_path = Path(args.dataset_path)
    scene_dict = {}

    camera_list = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]
    for camera in camera_list:
        if args.scene is None:
            scenes = [x for x in os.listdir(os.path.join(dataset_path, camera)) if os.path.isdir(os.path.join(dataset_path, camera, x))]
        else:
            scenes = [args.scene] if args.scene in os.listdir(os.path.join(dataset_path, camera)) else []
        for scene in scenes:
            if scene in scene_dict.keys():
                scene_dict[scene].append(camera)
            else:
                scene_dict[scene] = [camera]

    with open(os.path.join(dataset_path, 'calib_data.json'), 'r') as f:
        calib_dict = json.load(f)

    for scene, camera_list in scene_dict.items():
        prepare_single(args, scene, camera_list, calib_dict)

if __name__ == '__main__':
    args = parse_args()
    run_im(args)