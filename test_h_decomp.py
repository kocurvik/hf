import numpy as np
import poselib


def test_homography_decomposition():
    t = np.random.randn(3, 1)
    t /= np.linalg.norm(t)
    n = np.random.randn(3, 1)
    n /= np.linalg.norm(n)

    A = np.random.randn(3, 3)
    R, _ = np.linalg.qr(A)

    print("GT Rt")
    print(np.column_stack([R, t]))

    focal = 600
    K = np.diag([focal, focal, 1.0])

    H = R + (t @ n.T)

    poses, normals = poselib.motion_from_homography(H)

    for i in range(len(poses)):
        print("Rt est:")
        print(poses[i].Rt)

    poses, normals = poselib.motion_from_homography(-H)

    for i in range(len(poses)):
        print("Rt est:")
        print(poses[i].Rt)


if __name__ == '__main__':
    test_homography_decomposition()