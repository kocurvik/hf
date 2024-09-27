import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from dataset_utils.data import is_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path')
    parser.add_argument('calib_path')

    return parser.parse_args()

def calibrate(images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = 30 * np.mgrid[0:8, 0:5].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for fname in tqdm(images):
        # print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    width = gray.shape[1]
    height = gray.shape[0]

    ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = \
        cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)

    # img = cv2.imread(images[0])
    # undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    # plt.imshow(undistorted_img[:, :, ::-1])
    # plt.show()

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"total error: {mean_error / len(objpoints)} px")

    return mtx, dist, width, height


def get_cam_dict(mtx, dist, width, height):
    d = {}
    d['focal'] = (mtx[0, 0] + mtx[1, 1]) / 2
    d['fx'] = mtx[0, 0]
    d['fy'] = mtx[1, 1]
    d['pp'] = mtx[:2, 2].tolist()
    d['K'] = mtx.tolist()
    d['distortion_coeffs'] = dist.tolist()
    d['width'] = width
    d['height'] = height
    return d


def main(args):
    dirs = [os.path.join(args.calib_path, x) for x in os.listdir(args.calib_path) 
            if os.path.isdir(os.path.join(args.calib_path, x))]

    calib_data = {}

    for dir_path in dirs:
        images = [os.path.join(dir_path, 'Calib', x) for x in os.listdir(os.path.join(dir_path, 'Calib')) if
                  is_image(x)]
        cam_dict = get_cam_dict(*calibrate(images))
        calib_data[os.path.basename(dir_path)] = cam_dict
        
    with open(args.json_path, 'w') as f:
        json.dump(calib_data, f, indent=4)

    print(f"Calibration data saved to {args.json_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)