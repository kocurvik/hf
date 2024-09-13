
import numpy as np
import poselib
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

def focal_voting(x1_1, x2_1, x1_2, x3_2, pp, iterations, inlier_ratio, distance_threshold):
    focals = poselib.focal_from_homography(x1_1, x2_1, x1_2, x3_2, pp, iterations, inlier_ratio, distance_threshold)
    try:
        kde = gaussian_kde(focals)

        no_samples = 1000
        samples = np.linspace(min(focals), max(focals), no_samples)
        probs = kde.evaluate(samples)
        max_idx = probs.argmax()

        # plt.plot(samples, probs)
        # plt.show()
        # print(samples[max_idx])
        focal = samples[max_idx]
    except Exception:
        focal = 1.0
    return focal