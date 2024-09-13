import argparse
import itertools
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


def get_area(pts):
    width = np.max(pts[:, 0]) - np.min(pts[:, 0])
    height = np.max(pts[:, 1]) - np.min(pts[:, 1])
    return width * height


def create_gt_h5(cameras, images, out_dir, args):
    exist = [os.path.exists(os.path.join(out_dir, f'{x}.h5')) for x in ['K', 'R', 'T']]
    if not False in exist and not args.recalc:
        print(f"GT info exists in {out_dir} - not creating it anew")
        return

    print(f"Writing GT info to {out_dir}")
    fK = h5py.File(os.path.join(out_dir, 'K.h5'), 'w')
    fR = h5py.File(os.path.join(out_dir, 'R.h5'), 'w')
    fT = h5py.File(os.path.join(out_dir, 'T.h5'), 'w')

    for img_id, img in images.items():
        camera = cameras[img.camera_id]
        name = img.name.split('.')[0].replace('/', '\\')
        q = img.qvec
        t = img.tvec
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        K = cam_to_K(camera)

        fR.create_dataset(name, shape=(3, 3), data=R)
        fT.create_dataset(name, shape=(3, 1), data=t.reshape(3,1))
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

def extract_features(img_dir_path, images, cameras, out_dir, args):
    # extractor = lightglue.SuperPoint(max_num_keypoints=2048).eval().cuda()
    extractor = get_extractor(args)
    out_path = os.path.join(out_dir, f"{get_matcher_string(args)}.pt")


    if os.path.exists(out_path) and not args.recalc:
        print(f"Features already found in {out_path}")
        return

    print("Extracting features")
    feature_dict = {}

    for img_id, img in tqdm(images.items()):
        img_path = os.path.join(img_dir_path, img.name)
        name = img.name.split('.')[0]
        image_tensor = load_image(img_path).cuda()
        cam = cameras[img.camera_id]

        if cam.width != image_tensor.size(-1):
            if cam.width == image_tensor.size(-2):
                image_tensor = torch.swapaxes(image_tensor, -2, -1)
            else:
                print(f"Image dimensions do not comply with camera width and height for: {img_path} - skipping!")
                continue

        kp_tensor = extractor.extract(image_tensor, resize=args.resize)
        feature_dict[name] = kp_tensor

    torch.save(feature_dict, out_path)
    print("Features saved to: ", out_path)


def get_overlap_areas(cameras, images, pts, img_ids):
    img_id1, img_id2, img_id3 = img_ids
    imgs = list(images[x] for x in img_ids)
    cam_1, cam_2, cam_3 = (cameras[x.camera_id] for x in imgs)
    img_1, img_2, img_3 = imgs

    img_1_point3D_ids = np.array(img_1.point3D_ids)
    img_1_point3D_ids = img_1_point3D_ids[img_1_point3D_ids != -1]
    img_2_point3D_ids = np.array(img_2.point3D_ids)
    img_2_point3D_ids = img_2_point3D_ids[img_2_point3D_ids != -1]
    img_3_point3D_ids = np.array(img_3.point3D_ids)
    img_3_point3D_ids = img_3_point3D_ids[img_3_point3D_ids != -1]

    overlap = set(img_1_point3D_ids).intersection(set(img_2_point3D_ids)).intersection(img_3_point3D_ids)

    if len(overlap) < 5:
        return 0.0, 0.0, 0.0

    pts_img_1 = []
    pts_img_2 = []
    pts_img_3 = []

    for pt_id in list(overlap):
        pt = pts[pt_id]
        if img_id1 in pt.image_ids and img_id2 in pt.image_ids and img_id3 in pt.image_ids:
            idx1 = np.where(pt.image_ids == img_id1)[0][0]
            idx2 = np.where(pt.image_ids == img_id2)[0][0]
            idx3 = np.where(pt.image_ids == img_id3)[0][0]

            im_idx1 = pt.point2D_idxs[idx1]
            im_idx2 = pt.point2D_idxs[idx2]
            im_idx3 = pt.point2D_idxs[idx3]

            pts_img_1.append(img_1.xys[im_idx1])
            pts_img_2.append(img_2.xys[im_idx2])
            pts_img_3.append(img_3.xys[im_idx3])

    pts_img_1 = np.array(pts_img_1)
    pts_img_2 = np.array(pts_img_2)
    pts_img_3 = np.array(pts_img_3)

    area_1 = get_area(pts_img_1) / (cam_1.width * cam_1.height)
    area_2 = get_area(pts_img_2) / (cam_2.width * cam_2.height)
    area_3 = get_area(pts_img_3) / (cam_3.width * cam_3.height)

    return area_1, area_2, area_3


def create_triplets(out_dir, cameras, images, pts, args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = 0

    features = torch.load(os.path.join(out_dir, f"{get_matcher_string(args)}.pt"))

    matcher = lightglue.LightGlue(features=args.features).eval().cuda()

    if args.area is not None:
        h5_path_str = f'triplets-a{args.area}-{get_matcher_string(args)}-LG.h5'
    else:
        h5_path_str = f'triplets-{get_matcher_string(args)}-LG.h5'
    h5_path = os.path.join(out_dir, h5_path_str)
    h5_file = h5py.File(h5_path, 'w')
    print("Writing matches to: ", h5_path)

    triplets = []

    id_list = list([k for k, v in images.items()])
    inverse_id_list = {v.name.split('.')[0]: k for k,v in images.items()}

    if args.load:
        triples_txt_path = os.path.join(out_dir, f'triplets-{get_matcher_string(args)}-LG.txt')
        print("Using previous triplets from:", triples_txt_path)
        with open(triples_txt_path, 'r') as f:
            loaded_triplets = f.readlines()

        total = len(loaded_triplets)
        img_ids_list = []
        for triplet in loaded_triplets:
            name_1, name_2, name_3 = triplet.strip().split(' ')
            img_ids_list.append((inverse_id_list[name_1.split('.')[0]],
                                 inverse_id_list[name_2.split('.')[0]],
                                 inverse_id_list[name_3.split('.')[0]]))
    elif args.num_samples is None:
        img_ids_list = list(itertools.combinations(id_list, 3))
        total = len(img_ids_list)
    else:
        total = args.num_samples

    all_counter = 0

    with tqdm(total=total) as pbar:
        while output < total:
            if args.num_samples is not None:
                img_ids = random.sample(list(images.keys()), 3)
            else:
                if all_counter >= len(img_ids_list):
                    break
                img_ids = img_ids_list[all_counter]
                all_counter += 1
                pbar.update(1)

            label = '-'.join([images[x].name.split('.')[0] for x in img_ids])

            if label in h5_file:
                continue

            min_area = 0.1 if args.area is None else args.area
            area_1, area_2, area_3 = get_overlap_areas(cameras, images, pts, img_ids)
            if area_1 > min_area and area_2 > min_area and area_3 > min_area:
                img_1, img_2, img_3 = (images[x] for x in img_ids)

                feats_1 = features[img_1.name.split(".")[0]]
                feats_2 = features[img_2.name.split(".")[0]]
                feats_3 = features[img_3.name.split(".")[0]]

                out_12 = matcher({'image0': feats_1, 'image1': feats_2})
                out_13 = matcher({'image0': feats_1, 'image1': feats_3})
                out_23 = matcher({'image0': feats_2, 'image1': feats_3})

                scores_12 = out_12['matching_scores0'][0].detach().cpu().numpy()
                scores_13 = out_13['matching_scores0'][0].detach().cpu().numpy()
                scores_23 = out_23['matching_scores0'][0].detach().cpu().numpy()

                matches_12 = out_12['matches0'][0].detach().cpu().numpy()
                matches_13 = out_13['matches0'][0].detach().cpu().numpy()
                matches_23 = out_23['matches0'][0].detach().cpu().numpy()

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

                out_array = np.empty([len(idxs), 9])

                for i, x in enumerate(idxs):
                    idx_1, idx_2, idx_3 = x
                    point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
                    point_2 = feats_2['keypoints'][0, idx_2].detach().cpu().numpy()
                    point_3 = feats_3['keypoints'][0, idx_3].detach().cpu().numpy()
                    score_12 = scores_12[idx_1]
                    score_13 = scores_13[idx_1]
                    score_23 = scores_23[idx_2]
                    out_array[i] = np.array([*point_1, *point_2, *point_3, score_12, score_13, score_23])

                if len(idxs) < 10:
                    continue

                h5_file.create_dataset(label, shape=out_array.shape, data=out_array)
                triplets.append(label.replace('-', ' '))

                if args.num_samples is not None:
                    pbar.update(1)
                    output += 1

    if args.area is None:
        triples_txt_path = os.path.join(out_dir, f'triplets-{get_matcher_string(args)}-LG.txt')
    else:
        triples_txt_path = os.path.join(out_dir, f'triplets-a{args.area}-{get_matcher_string(args)}-LG.txt')
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
    dataset_path = Path(args.dataset_path)
    basename = os.path.basename(dataset_path)

    img_path, model_path, subset_path = get_dataset_paths(basename, dataset_path, subset)

    cameras, images, points = read_model(model_path)

    out_dir = os.path.join(args.out_path, subset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    create_gt_h5(cameras, images, out_dir, args)
    if 'loftr' in args.features:
        create_triplets_loftr(out_dir, img_path, cameras, images, points, args)
    else:
        extract_features(img_path, images, cameras, out_dir, args)
        create_triplets(out_dir, cameras, images, points, args)


    # gen = cam_pair_generator(matcher, subset_path, img_path, cameras, images, points, max_pairs=num_samples)
    # samples = [s for s in tqdm(gen, total=num_samples)]
    #
    # if not os.path.exists('saved/{}'.format(args.matcher)):
    #     os.makedirs('saved/{}'.format(args.matcher))
    # joblib.dump(samples, 'saved/{}/im_{}_{}.joblib'.format(args.matcher, basename, subset))


def get_dataset_paths(basename, dataset_path, subset):
    subset_path = os.path.join(dataset_path, subset)
    if basename.lower() == 'phototourism':
        model_path = os.path.join(subset_path, 'dense', 'sparse')
        img_path = os.path.join(subset_path, 'dense', 'images')
    elif basename.lower() == 'urban':
        model_path = os.path.join(subset_path, 'sfm')
        img_path = os.path.join(subset_path, 'images_full_set')
    elif 'aachen' in basename.lower():
        model_path = os.path.join(subset_path, '3D-models/aachen_v_1_1')
        img_path = os.path.join(subset_path, 'images_upright')
    elif 'multiview_undistorted' in basename.lower() or 'eth3d' in basename.lower():
        model_path = os.path.join(subset_path, 'dslr_calibration_undistorted')
        img_path = os.path.join(subset_path, 'images')
    elif 'cambridge' in basename.lower():
        model_path = os.path.join(subset_path, 'model_train')
        img_path = os.path.join(subset_path, subset)
    else:
        model_path = os.path.join(subset_path, 'sfm')
        img_path = os.path.join(subset_path, 'images_full')
    return img_path, model_path, subset_path


def run_im(args):
    dataset_path = Path(args.dataset_path)
    dir_list = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]

    for subset in dir_list:
        prepare_single(args, subset)

if __name__ == '__main__':
    args = parse_args()
    run_im(args)