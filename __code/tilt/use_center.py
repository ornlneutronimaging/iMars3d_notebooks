import logging
import numpy as np
from skimage import feature
import time

from .base_algorithm import BaseAlgorithm


class UseCenter(BaseAlgorithm):

    def compute(self, **kwds):

        centers = np.array(
                    [item for item
                     in self.iteration_centers(self.image_0_degree, self.image_180_degree, **kwds)])

        rows, centers = centers.T
        cm = np.median(centers)
        csigma = np.std(centers)
        w = (centers > cm - 1.5 * csigma) * (centers < cm + 1.5 * csigma)
        rows1 = rows[w]
        centers1 = centers[w]
        from scipy import stats
        try:
            tilt, intercept, r, p, std_err = stats.linregress(rows1, centers1)
        except ValueError:
            return np.NaN

        return tilt

    def iteration_centers(self, img0, img180, sigma=3, maxshift=20):
        edge0 = self.getEdge(img0, sigma=sigma)
        edge180 = self.getEdge(img180, sigma=sigma)
        edge180 = edge180[:, ::-1]
        for i, (line0, line180) in enumerate(zip(edge0, edge180)):
            c = self.compute_center_of_rotation(line0, line180, maxshift=maxshift)
            # print i,c
            if c > (line0.size - maxshift) / 2. + maxshift // 40.:  # remove edge cases
                yield i, c
            continue
        return

    def getEdge(self, img, **kwds):
        edge = feature.canny(img, **kwds)
        edge = np.array(edge, dtype="float32")
        return edge

    def compute_center_of_rotation(self, x1, x2, **kwds):
        shift = UseCenter.compute_shift(x1, x2, **kwds)
        return (shift + x1.size) / 2.

    @staticmethod
    def compute_shift(x1, x2, maxshift=20):
        """compute shift between two spectra
        when x1 is shifted by the result pixels, x1 is most similar to x2
        """
        diffs = []
        for dx in range(1 - maxshift, maxshift):
            if dx > 0:
                diff = x1[dx:] * x2[:-dx]
            elif dx < 0:
                diff = x1[:dx] * x2[-dx:]
            else:
                diff = x1 * x2
            diff = np.sum(diff * diff)
            diffs.append((dx, diff))
        diffs = np.array(diffs)
        # np.save("diffs.npy", diffs)
        X, Y = diffs.T
        w = np.argmax(Y)
        return X[w]
