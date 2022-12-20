import logging
import numpy as np
from skimage.transform import rotate
import time

from .base_algorithm import BaseAlgorithm

MAX_SHIFT = 400


class DirectMinimization(BaseAlgorithm):

    def compute(self):
        logging.info("Running 'direct minimization' tilt calculation ...")

        flipped_img180 = np.fliplr(self.image_180_degree)
        shift = self.find_shift(self.image_0_degree, flipped_img180)
        tilts = np.arange(-2., 2.1, 0.2)
        tilt = self.argmin_tilt(tilts, self.image_0_degree, flipped_img180, shift)
        tilts = np.arange(tilt - 0.2, tilt + 0.21, 0.02)
        tilt = self.argmin_tilt(tilts, self.image_0_degree, flipped_img180, shift)

        return tilt

    def shift_diff(self, x, img1, img2):
        # shift positive means img2 was shifted to the left,
        # or img1 was shifted to the right.
        x = int(x)
        if x > 0:
            left = img1[:, :-x]
            right = img2[:, x:]
        elif x < 0:
            left = img1[:, -x:]
            right = img2[:, :x]
        else:
            left = img1
            right = img2
        return left - right

    def shift_diff2(self, x, img1, img2):
        d = self.shift_diff(x, img1, img2)
        return (d ** 2).sum() / d.size

    def find_shift(self, img0, flipped_img180):
        """find the shift in number of pixels

        note: the relation between rot center and shift is
          rot_center = -shift/2 if 0 is center of image
        """
        # print("* Calculating shift...")
        ncols = img0.shape[1]

        def shift_diff2(x, img1, img2):
            d = self.shift_diff(x, img1, img2)
            return (d ** 2).sum() / d.size

        def diff(x):
            return shift_diff2(x, img0, flipped_img180)

        start = max(-ncols // 2, - MAX_SHIFT)
        end = min(MAX_SHIFT, ncols // 2)
        xs = range(start, end)
        diffs = [diff(x) for x in xs]
        index = np.argmin(diffs)
        guess = xs[index]
        return guess

    def argmin_tilt(self, tilts, img0, flipped_img180, shift):
        diffs = []
        for tilt in tilts:
            diff = self.shift_tilt_diff(shift, tilt, img0, flipped_img180)
            diff = np.sum(diff ** 2) / diff.size
            diffs.append(diff)
            continue
        return tilts[np.argmin(diffs)]

    def shift_tilt_diff(self, shift, tilt, img1, img2):
        nrows, ncols = img1.shape
        borderY, borderX = nrows // 20, ncols // 20
        a = rotate(img1 / np.max(img1), -tilt)[borderY:-borderY, borderX:-borderX]
        b = rotate(img2 / np.max(img2), tilt)[borderY:-borderY, borderX:-borderX]
        return self.shift_diff(shift, a, b)
