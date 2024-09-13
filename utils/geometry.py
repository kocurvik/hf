import numpy as np

def skew(x):
    x0, x1, x2 = x

    return np.array([[0, -x2, x1],
                     [x2, 0, -x0],
                     [-x1, x0, 0]])

def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1)))

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def get_pose(img1, img2, R_dict, T_dict):
    R1 = np.array(R_dict[img1])
    R2 = np.array(R_dict[img2])
    t1 = np.array(T_dict[img1])
    t2 = np.array(T_dict[img2])

    R = R2 @ R1.T
    t = -R @ t1 + t2
    return R, t


def get_gt_F(img1, img2, R_dict, T_dict, camera_dicts):
    R1, t1 = np.array(R_dict[img1]), np.array(T_dict[img1])
    R2, t2 = np.array(R_dict[img2]), np.array(T_dict[img2])
    R = R2 @ R1.T
    t = (t2 - R @ t1).ravel()
    cam1 = camera_dicts[img1]
    cam2 = camera_dicts[img2]
    K1 = np.array([[cam1['params'][0], 0.0, cam1['params'][-2]], [0.0, cam1['params'][0], cam1['params'][-1]], [0, 0, 1.0]])
    K2 = np.array([[cam2['params'][0], 0.0, cam2['params'][-2]], [0.0, cam2['params'][0], cam2['params'][-1]], [0, 0, 1.0]])

    return np.linalg.inv(K2.T) @ skew(t) @ R @ np.linalg.inv(K1)


def get_gt_E(img1, img2, R_dict, T_dict, camera_dicts):
    R1, t1 = np.array(R_dict[img1]), np.array(T_dict[img1])
    R2, t2 = np.array(R_dict[img2]), np.array(T_dict[img2])
    R = R2 @ R1.T
    t = (t2 - R @ t1).ravel()

    return skew(t) @ R


def get_inliers(F, x1, x2, threshold = 2.0):
    pts1 = np.column_stack([x1, np.ones(len(x1))])
    pts2 = np.column_stack([x2, np.ones(len(x2))])
    F_t = F.T
    line1_in_2 = pts1 @ F_t
    line2_in_1 = pts2 @ F

    # numerator = (x'^T F x) ** 2
    numerator = np.sum(pts2 * line1_in_2, axis=1) ** 2

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator = line1_in_2[:, 0] ** 2 + line1_in_2[:, 1] ** 2 + line2_in_1[:, 0] ** 2 + line2_in_1[:, 1] ** 2
    out = numerator / denominator
    return out < threshold ** 2


def add_rand_pts(x, cam_dict, multiplier):
    x_new = np.random.rand(int(multiplier * len(x)), 2)
    x_new[:, 0] *= cam_dict['width']
    x_new[:, 1] *= cam_dict['height']
    return np.row_stack([x, x_new])


def force_inliers(x1, x2, x3, img1, img2, img3, R_dict, T_dict, camera_dicts, ratio):
    F12 = get_gt_F(img1, img2, R_dict, T_dict, camera_dicts)
    F13 = get_gt_F(img1, img3, R_dict, T_dict, camera_dicts)
    F23 = get_gt_F(img2, img3, R_dict, T_dict, camera_dicts)

    inliers_12 = get_inliers(F12, x1, x2, 2.0)
    inliers_13 = get_inliers(F13, x1, x3, 2.0)
    inliers_23 = get_inliers(F23, x2, x3, 2.0)

    l = np.logical_and(np.logical_and(inliers_12, inliers_13), inliers_23)

    # print(np.sum(inliers_12), np.sum(inliers_13), np.sum(inliers_23), np.sum(l), len(x1))

    multiplier = (1 - ratio) / ratio

    x1, x2, x3 = x1[l], x2[l], x3[l]

    x1 = add_rand_pts(x1, camera_dicts[img1], multiplier)
    x2 = add_rand_pts(x2, camera_dicts[img2], multiplier)
    x3 = add_rand_pts(x3, camera_dicts[img3], multiplier)

    return x1, x2, x3