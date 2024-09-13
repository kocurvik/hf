import numpy as np
import poselib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from utils.geometry import skew, angle, rotation_angle


def generate_points(num_pts, f, distance=2, depth=1, dominant_plane=0.0,  width=640, height=480):
    num_pts_plane = int(dominant_plane * num_pts)
    num_pts_other = num_pts - num_pts_plane
    zs = (1 + distance) * f + depth * f * np.ones(num_pts_plane)# * np.random.rand(num_pts)
    zs_2 = (1 + distance) * f + depth * f * np.random.randn(num_pts_other)
    zs = np.concatenate([zs, zs_2])

    xs = (np.random.rand(num_pts) - 0.5) * width * (1 + distance)
    ys = (np.random.rand(num_pts) - 0.5) * height * (1 + distance)
    # X = np.column_stack([xs, ys, zs])
    # X = (Rotation.from_euler('xyz', (0, 10, 0), degrees=True).as_matrix() @ X.T).T
    # return np.column_stack([X, np.ones_like(xs)])
    return np.column_stack([xs, ys, zs, np.ones_like(xs)])


def get_projection(P, X):
    x = P @ X.T
    x = x[:2, :] / x[2, np.newaxis, :]
    return x.T

def visible_in_view(x, width=640, height=480):
    visible = np.logical_and(np.abs(x[:, 0]) <= width / 2, np.abs(x[:, 1]) <= height / 2)
    return visible


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_scene(points, R, t, f, width=640, height=480, color_1='black', color_2='red', name=""):
    c_x_1 = np.array([0.5 * width, 0.5 * width, -0.5 * width, -0.5 * width, 0])
    c_y_1 = np.array([0.5 * height, -0.5 * height, -0.5 * height, 0.5 * height, 0])
    c_z_1 = np.array([f, f, f, f, 0])
    c_z_2 = np.array([f, f, f, f, 0])

    camera2_X = np.row_stack([c_x_1, c_y_1, c_z_2, np.ones_like(c_x_1)])
    c_x_2, c_y_2, c_z_2 = np.column_stack([R.T, -R.T @ t]) @ camera2_X

    # fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.set_box_aspect([1.0, 1., 1.0])

    ax.plot3D(c_x_1, c_y_1, c_z_1, color_1)
    ax.plot3D(c_x_2, c_y_2, c_z_2, color_2)

    ax.plot3D([c_x_1[0], c_x_1[3]], [c_y_1[0], c_y_1[3]], [c_z_1[0], c_z_1[3]], color_1)
    ax.plot3D([c_x_2[0], c_x_2[3]], [c_y_2[0], c_y_2[3]], [c_z_2[0], c_z_2[3]], color_2)

    for i in range(4):
        ax.plot3D([c_x_1[i], c_x_1[-1]], [c_y_1[i], c_y_1[-1]], [c_z_1[i], c_z_1[-1]], color_1)
        ax.plot3D([c_x_2[i], c_x_2[-1]], [c_y_2[i], c_y_2[-1]], [c_z_2[i], c_z_2[-1]], color_2)

    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='blue')

    set_axes_equal(ax)

    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(name)


def get_scene(f, R1, t1, R2, t2, num_pts, X=None, seed=None, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    K = np.diag([f, f, 1])


    P1 = K @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K @ np.column_stack([R1, t1])
    P3 = K @ np.column_stack([R2, t2])

    if X is None:
        X = generate_points(num_pts, f, **kwargs)
    x1 = get_projection(P1, X)
    x2 = get_projection(P2, X)
    x3 = get_projection(P3, X)

    # visible_2 = visible_in_view(x2, width=width, height=height)
    # visible_3 = visible_in_view(x3, width=width, height=height)
    # x1, x2, X = x1[visible][:num_pts], x2[visible][:num_pts], X[visible]

    return x1, x2, x3, X


def run_synth():
    f = 600
    R12 = Rotation.from_euler('xyz', (3, 60, 0), degrees=True).as_matrix()
    R13 = Rotation.from_euler('xyz', (-5, -30, 0), degrees=True).as_matrix()
    c1 = np.array([2 * f, 0, f])
    c2 = np.array([0, f, 0.5 * f])
    # R = Rotation.from_euler('xyz', (theta, 30, 0), degrees=True).as_matrix()
    # c = np.array([f1, y, 0])
    t12 = -R12 @ c1
    t13 = -R13 @ c2
    # t13 = 2 * t13 * np.linalg.norm(t12) / np.linalg.norm(t13)

    x1, x2, x3, X = get_scene(f, R12, t12, R13, t13, 100, dominant_plane=0.8)

    sigma = 2.5

    x1 += sigma * np.random.randn(*(x1.shape))
    x2 += sigma * np.random.randn(*(x1.shape))
    x3 += sigma * np.random.randn(*(x1.shape))

    # idxs1 = np.random.permutation(np.arange(30))
    # x1[:30] = x1[idxs1]
    # idxs2 = np.random.permutation(np.arange(30, 60))
    # x2[30:60] = x2[idxs2]


    T12 = np.diag([0, 0, 0, 1.0])
    T12[:3, :3] = R12
    T12[:3, 3] = t12
    T13 = np.diag([0, 0, 0, 1.0])
    T13[:3, :3] = R13
    T13[:3, 3] = t13

    T23 = T13 @ np.linalg.inv(T12)
    R23 = T23[:3, :3]
    t23 = T23[:3, 3]

    # print("T12")
    # print(T12)
    # print("T13")
    # print(T13)

    ransac_dict = {'max_epipolar_error': 2.5, 'progressive_sampling': False,
                   'min_iterations': 1, 'max_iterations': 100, 'lo_iterations': 0,
                   'inner_refine': False, 'threeview_check': True, 'use_homography': True, 'scaled_relpose': True,
                   'use_hc': False}

    pp = np.array([0, 0])
    # out, info = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, pp, ransac_dict, {'max_iterations': 0, 'verbose': False})
    # pose = out.poses

    focals = poselib.focal_from_homography(x1, x2, x1, x3, pp, 100, 0.2, 5.0)

    print(focals)
    print(len(focals))
    print(len(np.unique(focals)))

    # print("R errs")
    # print(rotation_angle(pose.pose12.R.T @ R12))
    # print(rotation_angle(pose.pose13.R.T @ R13))
    # print(rotation_angle(pose.pose23().R.T @ R23))
    # #
    # print("T errs")
    # print(angle(pose.pose12.t, t12))
    # print(angle(pose.pose13.t, t13))
    # print(angle(pose.pose23().t, t23))
    #
    # print("f err")
    # f_err = np.abs(f - out.camera.focal()) / f
    # print(f_err)
    # print("Info")
    # print(f'iterations: {info["iterations"]} \t num_inliers: {info["num_inliers"] / len(x1)}')

    # inlier_ratio = info["num_inliers"] / len(x1)

    # return out5['iterations'], out4['iterations'], outR['iterations']
    inlier_ratio = 0
    f_err = np.abs(np.nanmedian(focals) - f) / f
    return f_err, inlier_ratio


if __name__ == '__main__':
    iters = [run_synth() for _ in range(1)]
    f_errs = [x[0] for x in iters]
    inlier_ratios = [x[1] for x in iters]

    print(f"Mean f_err: {np.mean(f_errs)} - Median f_err: {np.nanmedian(f_errs)}")
    print(f"Mean IR: {np.mean(inlier_ratios)} - Median IR: {np.nanmedian(inlier_ratios)}")
