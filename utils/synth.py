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

    return np.column_stack([xs, ys, zs, np.ones_like(xs)])


def get_projection(P, X):
    x = P @ X.T
    x = x[:2, :] / x[2, np.newaxis, :]
    return x.T

def visible_in_view(x, width=640, height=480, **kwargs):
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
    return ax


def look_at_rotation(camera_center, target_point):
    """
    Computes the rotation matrix that aligns the camera's optical axis
    to point at a specific target in 3D space.

    :param camera_center: The 3D coordinates of the camera center (a point in world coordinates).
    :param target_point: The 3D coordinates of the point the camera should look at.
    :return: A 3x3 rotation matrix that rotates the camera to look at the target point.
    """
    # Compute the forward vector (from the camera center to the target point)
    forward = target_point - camera_center
    forward = forward / np.linalg.norm(forward)  # Normalize the forward vector

    # Assume that the up vector of the camera is the world y-axis (0, 1, 0)
    world_up = np.array([0, -1, 0])

    # Compute the right vector by taking the cross product of forward and world up
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        # If forward and world_up are nearly collinear, use a different up vector
        world_up = np.array([1, 0, 0])
        right = np.cross(world_up, forward)

    right = right / np.linalg.norm(right)  # Normalize the right vector

    # Compute the true up vector (orthogonal to both forward and right)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # Create the rotation matrix: [right, up, -forward] (column-wise)
    # Camera looks along the -z axis (forward vector)
    rotation_matrix = np.stack([right, up, forward], axis=1)

    return rotation_matrix


def get_random_scene(f1, f2, f3, n, dominant_plane=0.5, width=1920, height=1080):
    look_at2 = 4 * f1 * (0.5 - np.random.rand(3)) + np.array([0, 0, 5 * f1])
    c2 = 4 * f1 * (0.5 - np.random.rand(3))
    R2 = look_at_rotation(c2, look_at2)
    c3 = 4 * f1 * (0.5 - np.random.rand(3))
    look_at3 = 4 * f1 * (0.5 - np.random.rand(3)) + np.array([0, 0, 5 * f1])
    R3 = look_at_rotation(c3, look_at3)
    t2 = -R2 @ c2
    t3 = -R3 @ c3

    K1 = np.diag([f1, f1, 1])
    K2 = np.diag([f2, f2, 1])
    K3 = np.diag([f3, f3, 1])

    P1 = K1 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K2 @ np.column_stack([R2, t2])
    P3 = K3 @ np.column_stack([R3, t3])

    num_plane = int(dominant_plane * n)
    num_pts = n - num_plane

    X = 4 * f1 * (0.5 - np.random.rand(10 * num_pts, 3)) + np.array([0, 0, 5 * f1])
    X_h = np.column_stack([X, np.ones(len(X))])
    x1 = get_projection(P1, X_h)
    x2 = get_projection(P2, X_h)
    x3 = get_projection(P3, X_h)

    l = visible_in_view(x1, width=width, height=height) + visible_in_view(x2, width=width, height=height) + visible_in_view(x3, width=width, height=height)

    idxs = np.where(l)[0][:num_pts]
    if len(idxs) < num_pts:
        return get_random_scene(f1, f2, f3, n, width=width, height=height)

    x1, x2, x3, X = x1[idxs], x2[idxs], x3[idxs], X[idxs]

    plane_angle_x = 30 / 180 * np.pi * np.random.randn()
    plane_angle_y = 30 / 180 * np.pi * np.random.randn()

    Xp = 4 * f1 * (0.5 - np.random.rand(10 * num_plane, 3))
    Xp[:, 2] = np.tan(plane_angle_x) * Xp[:, 0] + np.tan(plane_angle_y) * Xp[:, 1] + 5 * f1
    Xp_h = np.column_stack([Xp, np.ones(len(Xp))])
    x1p = get_projection(P1, Xp_h)
    x2p = get_projection(P2, Xp_h)
    x3p = get_projection(P3, Xp_h)

    visible_p = visible_in_view(x1p, width=width, height=height) + visible_in_view(x2p, width=width, height=height) + visible_in_view(x3p, width=width, height=height)

    idxs = np.where(visible_p)[0][:num_plane]
    if len(idxs) < num_plane:
        return get_random_scene(f1, f2, f3, n, width=width, height=height)

    x1p, x2p, x3p, Xp = x1p[idxs], x2p[idxs], x3p[idxs], Xp[idxs]

    x1 = np.row_stack([x1, x1p])
    x2 = np.row_stack([x2, x2p])
    x3 = np.row_stack([x3, x3p])
    X = np.row_stack([X, Xp])

    # ax = plot_scene(X, R2, t2, f1, width=width, height=height)
    # ax.scatter3D(*look_at2, color='g')
    # plt.show()

    return x1, x2, x3, X


def get_scene(f1, f2, f3, R1, t1, R2, t2, num_pts, X=None, seed=None, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    K1 = np.diag([f1, f1, 1])
    K2 = np.diag([f2, f2, 1])
    K3 = np.diag([f3, f3, 1])


    P1 = K1 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K2 @ np.column_stack([R1, t1])
    P3 = K3 @ np.column_stack([R2, t2])

    if X is None:
        X = generate_points(3 * num_pts, f1, **kwargs)
    x1 = get_projection(P1, X)
    x2 = get_projection(P2, X)
    x3 = get_projection(P3, X)

    # visible_2 = visible_in_view(x2, **kwargs)
    # visible_3 = visible_in_view(x3, **kwargs)
    # visible = np.logical_and(visible_2, visible_3)
    #
    # x1 = x1[visible]
    # x2 = x2[visible]
    # x3 = x3[visible]
    # X = X[visible]

    return x1, x2, x3, X


def run_synth():
    f1 = 1600
    R12 = Rotation.from_euler('xyz', (0, 0, 0), degrees=True).as_matrix()
    R13 = Rotation.from_euler('xyz', (0, 0, 0), degrees=True).as_matrix()
    c1 = np.array([2 * f1, 0, f1])
    c2 = np.array([0, f1, 0.5 * f1])
    # R = Rotation.from_euler('xyz', (theta, 30, 0), degrees=True).as_matrix()
    # c = np.array([f1, y, 0])
    t12 = -R12 @ c1
    t13 = -R13 @ c2
    # t13 = 2 * t13 * np.linalg.norm(t12) / np.linalg.norm(t13)

    f2 = f1
    f3 = 1200

    x1, x2, x3, X = get_scene(f1, f2, f3, R12, t12, R13, t13, 100, dominant_plane=0.8)
    # x1, x2, x3, X = get_random_scene(f1, f2, f3, 100, dominant_plane=0.8)

    sigma = 1.5

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
                   'min_iterations': 1000, 'max_iterations': 1000, 'lo_iterations': 25,
                   'use_homography': True, 'use_degensac': False}

    pp = np.array([0, 0])

    # out, info = poselib.estimate_shared_focal_relative_pose(x1, x2, pp, ransac_dict, {'max_iterations': 0, 'verbose': False})
    # camera2 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1, 'params': [f3, 0, 0]}
    # out, info = poselib.estimate_onefocal_relative_pose(x1, x3, camera2, pp, ransac_dict, {'max_iterations': 0, 'verbose': False})
    # focal = out.camera1.focal()
    # print(focal)
    # print(out.pose.R)
    # print(rotation_angle(out.pose.R.T @ R12))
    # print(angle(out.pose.t, t12))


    ransac_dict['use_homography'] = True
    ransac_dict['use_degensac'] = False
    ransac_dict['use_onefocal'] = True
    camera3 = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1, 'params': [f3, 0, 0]}
    # out, info = poselib.estimate_three_view_case2_relative_pose(x1, x2, x3, camera3, pp, ransac_dict, {'max_iterations': 100, 'verbose': False})
    out, info = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, pp, ransac_dict,
                                                            {'max_iterations': 0, 'verbose': False})
    pose = out.poses



    # print("R errs")
    print(rotation_angle(pose.pose12.R.T @ R12))
    print(rotation_angle(pose.pose13.R.T @ R13))
    print(rotation_angle(pose.pose23().R.T @ R23))
    #
    # print("T errs")
    print(angle(pose.pose12.t, t12))
    print(angle(pose.pose13.t, t13))
    print(angle(pose.pose23().t, t23))

    f1_err = np.abs(f1 - out.camera1.focal()) / f1
    print("f1 err: ", f1_err)

    f3_err = np.abs(f3 - out.camera3.focal()) / f3
    print("f3 err: ", f3_err)


    print("Info")
    print(f'iterations: {info["iterations"]} \t num_inliers: {info["num_inliers"] / len(x1)}')

    inlier_ratio = info["num_inliers"] / len(x1)

    return f1_err, inlier_ratio


if __name__ == '__main__':
    get_random_scene(2000, 2000, 2000, 300, width=1920, height=1080)


    iters = [run_synth() for _ in range(1)]
    f_errs = [x[0] for x in iters]
    inlier_ratios = [x[1] for x in iters]

    print(f"Mean f_err: {np.mean(f_errs)} - Median f_err: {np.nanmedian(f_errs)}")
    print(f"Mean IR: {np.mean(inlier_ratios)} - Median IR: {np.nanmedian(inlier_ratios)}")
