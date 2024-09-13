import numpy as np
import lightglue
from kornia.feature import LoFTR
import cv2
import torch

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

class LoFTRMatcher():
    matcher_str = 'loftr'

    def __init__(self, weights='outdoor', device='cuda', max_dim=512):
        self.loftr = LoFTR(weights).to(device)
        self.device = device
        self.max_dim = max_dim

    def enforce_dim(self, img):
        if self.max_dim is None:
            return img, 1.0

        h, w = img.shape[:2]

        gr = max(h, w)

        if gr > self.max_dim:
            scale_factor = self.max_dim / gr
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            return img, scale_factor
        else:
            return img, 1.0

    def match(self, img_1, img_2):
        img1_b = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img2_b = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        img1_b, s1 = self.enforce_dim(img1_b)
        img2_b, s2 = self.enforce_dim(img2_b)

        data = {'image0': frame2tensor(img1_b, self.device), 'image1': frame2tensor(img2_b, self.device)}
        pred = self.loftr(data)
        kp_1 = pred['keypoints0'].detach().cpu().numpy() / s1
        kp_2 = pred['keypoints1'].detach().cpu().numpy() / s2
        conf = pred['confidence'].detach().cpu().numpy()

        return conf, kp_1, kp_2