import argparse
import itertools
import json
import os
import random
from pathlib import Path

import cv2
import h5py
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import lightglue
from lightglue.utils import load_image, rbd

from utils.matching import LoFTRMatcher
from utils.read_write_colmap import cam_to_K, read_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=None)
    parser.add_argument('-a', '--area', type=float, default=None)
    parser.add_argument('-s', '--seed', type=int, default=100)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-f', '--features', type=str, default='superpoint')
    parser.add_argument('-mf', '--max_features', type=int, default=2048)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()

def create_gt_h5(images, calib, out_dir, args):
    exist = [os.path.exists(os.path.join(out_dir, f'{x}.h5')) for x in ['K', 'R', 'T']]
    if not False in exist and not args.recalc:
        print(f"GT info exists in {out_dir} - not creating it anew")
        return

    print(f"Writing GT info to {out_dir}")
    fK = h5py.File(os.path.join(out_dir, 'K.h5'), 'w')

    for name in images:
        cam_id = calib['images'][name]
        K = calib['cameras'][cam_id]['K']
        fK.create_dataset(name, shape=(3, 3), data=K)

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

def extract_features(img_dir_path, images, out_dir, args):
    extractor = get_extractor(args)
    out_path = os.path.join(out_dir, f"{get_matcher_string(args)}.pt")


    if os.path.exists(out_path) and not args.recalc:
        print(f"Features already found in {out_path}")
        return

    print("Extracting features")
    feature_dict = {}

    for img in tqdm(images):
        img_path = os.path.join(img_dir_path, img)
        image_tensor = load_image(img_path).cuda()

        kp_tensor = extractor.extract(image_tensor, resize=args.resize)
        feature_dict[img] = kp_tensor

    torch.save(feature_dict, out_path)
    print("Features saved to: ", out_path)


def create_triplets(out_dir, images, args, img_list=None):
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = 0

    features = torch.load(os.path.join(out_dir, f"{get_matcher_string(args)}.pt"))

    matcher = lightglue.LightGlue(features=args.features).eval().cuda()

    triplet_h5_path_str = f'triplets-{get_matcher_string(args)}-LG.h5'
    triplet_h5_path = os.path.join(out_dir, triplet_h5_path_str)
    triplet_h5_file = h5py.File(triplet_h5_path, 'w')
    print("Writing triplet matches to: ", triplet_h5_path)

    triplets = []

    if args.num_samples is None:
        if img_list is None:
            img_list = list(itertools.combinations(images, 3))
        total = len(img_list)
    else:
        total = args.num_samples

    all_counter = 0

    with tqdm(total=total) as pbar:
        while output < total:
            if args.num_samples is not None:
                img_triplet = random.sample(list(images.keys()), 3)
            else:
                if all_counter >= len(img_list):
                    break
                img_triplet = img_list[all_counter]
                all_counter += 1
                pbar.update(1)

            triplet_label = '-'.join(img_triplet)

            if triplet_label in triplet_h5_file:
                continue

            img_1, img_2, img_3 = img_triplet

            feats_1 = features[img_1]
            feats_2 = features[img_2]
            feats_3 = features[img_3]

            out_12 = matcher({'image0': feats_1, 'image1': feats_2})
            out_13 = matcher({'image0': feats_1, 'image1': feats_3})
            out_23 = matcher({'image0': feats_2, 'image1': feats_3})

            scores_12 = out_12['matching_scores0'][0].detach().cpu().numpy()
            scores_13 = out_13['matching_scores0'][0].detach().cpu().numpy()
            scores_23 = out_23['matching_scores0'][0].detach().cpu().numpy()

            matches_12 = out_12['matches0'][0].detach().cpu().numpy()
            matches_13 = out_13['matches0'][0].detach().cpu().numpy()
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
                point_2 = feats_2['keypoints'][0, idx_2].detach().cpu().numpy()
                point_3 = feats_3['keypoints'][0, idx_3].detach().cpu().numpy()
                score = scores_12[idx_1]
                score_13 = scores_13[idx_1]
                score_23 = scores_23[idx_2]
                out_triplet_array[i] = np.array([*point_1, *point_2, *point_3, score, score_13, score_23])

            if len(idxs) < 10:
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
                    point_2 = feats_2['keypoints'][0, idx_2].detach().cpu().numpy()
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
                    point_2 = feats_3['keypoints'][0, idx_2].detach().cpu().numpy()
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
                    point_1 = feats_2['keypoints'][0, idx_1].detach().cpu().numpy()
                    point_2 = feats_3['keypoints'][0, idx_2].detach().cpu().numpy()
                    score = scores_23[idx_1]
                    out_array.append(np.array([*point_1, *point_2, score]))
                out_array = np.array(out_array)
                triplet_h5_file.create_dataset(pair_label, shape=out_array.shape, data=out_array)


            if args.num_samples is not None:
                pbar.update(1)
                output += 1

    triples_txt_path = os.path.join(out_dir, f'triplets-{get_matcher_string(args)}-LG.txt')
    print("Writing list of triplets to: ", triples_txt_path)
    with open(triples_txt_path, 'w') as f:
        f.writelines(line + '\n' for line in triplets)

def read_loftr_image(img_dir_path, img, cameras):
    img_path = os.path.join(img_dir_path, img.name)
    image_array = cv2.imread(img_path)
    cam = cameras[img.camera_id]

    if cam.width != image_array.shape[1]:
        if cam.width == image_array.shape[0]:
            image_array = np.swapaxes(image_array, -2, -1)
        else:
            print(f"Image dimensions do not comply with camera width and height for: {img_path} - skipping!")
            return None
    return image_array

def create_triplets_loftr(out_dir, img_path, cameras, images, pts, args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = 0

    matcher = LoFTRMatcher(max_dim = args.resize, device='cuda')
    args.max_features = 0
    h5_path = os.path.join(out_dir, f'triplets-{get_matcher_string(args)}.h5')
    h5_file = h5py.File(h5_path, 'w')

    triplets = []

    print("Writing matches to: ", h5_path)

    with tqdm(total=args.num_samples) as pbar:
        while output < args.num_samples:
            img_ids = random.sample(list(images.keys()), 3)
            label = '-'.join([images[x].name.split('.')[0] for x in img_ids])

            if label in h5_file:
                continue

            area_1, area_2, area_3 = get_overlap_areas(cameras, images, pts, img_ids)
            if area_1 > 0.1 and area_2 > 0.1 and area_3 > 0.1:
                img_1, img_2, img_3 = (images[x] for x in img_ids)

                img_array_1 = read_loftr_image(img_path, img_1, cameras)
                img_array_2 = read_loftr_image(img_path, img_2, cameras)
                img_array_3 = read_loftr_image(img_path, img_3, cameras)

                if img_array_1 is None or img_array_2 is None or img_array_3 is None:
                    print("Noooo")
                    continue

                scores_12, kp_12_1, kp_12_2 = matcher.match(img_array_1, img_array_2)
                scores_13, kp_13_1, kp_13_3 = matcher.match(img_array_1, img_array_3)


                idxs = []

                for idx_12, kp in enumerate(kp_12_1):
                    idx_13 = np.where(np.all(kp_13_1==kp, axis=1))[0]

                    if len(idx_13) > 0:
                        idxs.append((idx_12, idx_13[0]))

                out_array = np.empty([len(idxs), 9])

                for i, x in enumerate(idxs):
                    idx_12, idx_13 = x
                    point_1 = kp_12_1[idx_12]
                    point_2 = kp_12_2[idx_12]
                    point_3 = kp_13_3[idx_13]
                    score_12 = scores_12[idx_12]
                    score_13 = scores_13[idx_13]

                    out_array[i] = np.array([*point_1, *point_2, *point_3, score_12, score_13, 0.0])

                h5_file.create_dataset(label, shape=out_array.shape, data=out_array)
                triplets.append(label.replace('-', ' '))
                pbar.update(1)
                output += 1

    if args.area is None:
        triples_txt_path = os.path.join(out_dir, f'triplets-{get_matcher_string(args)}.txt')
    else:
        triples_txt_path = os.path.join(out_dir, f'triplets-a{args.area}-{get_matcher_string(args)}.txt')
    print("Writing list of triplets to: ", triples_txt_path)
    with open(triples_txt_path, 'w') as f:
        f.writelines(line + '\n' for line in triplets)



def prepare_single(args, subset):
    out_dir = os.path.join(args.out_path, subset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    subset_path = os.path.join(args.dataset_path, subset)
    img_list = [x for x in os.listdir(subset_path) if '.jpg' in x.lower() or '.png' in x.lower()]

    calib_path = os.path.join(subset_path, 'calibration.json')
    with open(calib_path, 'r') as f:
        cameras = json.load(f)

    create_gt_h5(img_list, cameras, out_dir, args)

    triplet_list_path = os.path.join(subset_path, 'triplets.txt')
    if os.path.exists(triplet_list_path):
        with open(triplet_list_path, 'r') as f:
            triplets = [tuple(x.strip().split(' ')) for x in f.readlines()]
    else:
        triplets = None

    extract_features(subset_path, img_list, out_dir, args)
    create_triplets(out_dir, img_list, args, triplets)

def run_im(args):
    dataset_path = Path(args.dataset_path)
    dir_list = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x)) and '_' not in x]

    for subset in dir_list:
        prepare_single(args, subset)

if __name__ == '__main__':
    args = parse_args()
    run_im(args)