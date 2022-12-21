import logging
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
import time

from .base_algorithm import BaseAlgorithm
from .smooth import smooth


class PhaseCorrelation(BaseAlgorithm):

    border = 0.01
    rotation = 30.  # 45.
    # this is for collecting I(theta) histogram.
    # better be integer multiplication of 360
    bins = 360

    def compute(self):
        # implementation details:
        #  * the signal lines in the freq domain may overlap with
        #  x,y axes, where artifacts exist. so we rotate
        #  the input images by self.rotation

        data0 = self.image_0_degree
        hist0 = self.compute_intensity_theta_histogram(data0)

        data180 = self.image_180_degree
        data180_flipped = np.fliplr(data180)  # flip horizontally
        hist180 = self.compute_intensity_theta_histogram(data180_flipped)

        # correlate
        r = self._correlate(hist0, hist180)

        # find peak position
        tilt, weight = self._findPeakPosition(r)

        # make tilt center around 0
        if tilt > 180:
            tilt = tilt - 360

        # tilt is the rotation angle divided by 2
        # return tilt / 2, weight

        tilt_calculated = tilt / 2.

        return tilt_calculated

    def _correlate(self, hist0, hist180):
        # now that we have the histogram I(theta), we use
        # phase correlation method to determine the shift
        hist0 = smooth(hist0)[:hist0.size]
        hist180 = smooth(hist180)[:hist180.size]
        iq0 = np.fft.fft(hist0)
        iq180 = np.fft.fft(hist180)
        corr = iq180 * np.conjugate(iq0)
        corr /= np.abs(corr)
        r = np.fft.ifft(corr)
        r = np.real(r)
        return r

    def compute_intensity_theta_histogram(self, data):
        data0 = ndimage.rotate(data, self.rotation)
        # self._updateProgress()
        sizeY, sizeX = data0.shape
        # only take the rectangle region where data exist
        data0 = data0[sizeY // 4:sizeY * 3 // 4, sizeX // 4:sizeX * 3 // 4]
        # create histogram
        angles0, F0 = self.fft_angles_and_intensities(data0, self.border)
        hist0, edges0 = np.histogram(angles0, weights=F0, bins=self.bins)
        # remove points around 0, 90, 180, 270
        for index in [0, 90, 180, 270]:
            self.remove_badpoints(hist0, index, 5)
            continue
        return hist0

    def fft_angles_and_intensities(self, image, border):
        """read image and create the angles and intensities
        in the frequency domain for the image.
        """
        border = int(border * min(image.shape))

        F = np.fft.fft2(image)

        # clean up borders
        F[0:border, :] = np.nan
        F[:, 0:border] = np.nan
        F[-border:] = np.nan
        F[:, -border:] = np.nan

        # fill borders with zeros
        F[F != F] = 0
        # shift origin to the center of the freq-domain image
        F = np.fft.fftshift(F)

        # calculate angles
        sizey, sizex = F.shape
        x = np.arange(0, sizex, 1.) - sizex // 2
        xx = np.tile(x, sizey).reshape(F.shape)

        y = np.arange(0, sizey, 1.) - sizey // 2
        yy = np.repeat(y, sizex).reshape(F.shape)

        rho = (xx * xx + yy * yy) ** .5
        angles = np.arctan2(yy, xx)
        # clip along r axis
        R = min(image.shape) // 2
        bracket = (rho > 0.1 * R) * (rho < R)

        return angles[bracket], np.abs(F[bracket])

    def remove_badpoints(self, spectrum, index, width):
        assert width > 0
        good = np.median(
            np.concatenate((spectrum[index + width: index + 2 * width], spectrum[index - 2 * width: index - width])))
        start = index - width
        end = index + width
        if start * end > 0:
            ranges = (start, end),
        elif start < 0 and end > 0:
            ranges = (start, -1), (0, end)
        else:
            raise ValueError("Don't know how to deal with region (%s, %s)" % (start, end))

        # assign good values to bad points
        for s, e in ranges:
            spectrum[s:e] = good
            continue
        return

    def _findPeakPosition(self, r):
        # the argmax of r should be what we want.
        # - only data within a few degrees are useful
        sigma = np.std(r[1:])  # need this later
        # take the spectrum around 0 degree
        WIDTH_TO_SEARCH = 10
        r1 = np.concatenate((r[-WIDTH_TO_SEARCH:], r[:WIDTH_TO_SEARCH]))
        r1[WIDTH_TO_SEARCH] = 0  # set value at 0 degree to zero
        index = np.argmax(r1)
        # check if the max value is larger than fluctuation
        if r1[index] < 3.5 * sigma:
            return 0, np.exp(-(r1[index] - sigma) ** 2 / 2 / sigma / sigma)
        # - fit the peak with a polynomial and get the high point
        width = 2
        if index - width <= 0:
            # XXX need more checking of this
            return 0, np.exp(-(r1[index] - sigma) ** 2 / 2 / sigma / sigma)
        weight = 1 - np.exp(-r1[index] ** 2 / 2 / sigma / sigma)
        peak = r1[index - width: index + width + 1]

        def poly2(x, *p):
            a0, a1, a2 = p
            return a0 - a1 * (x - a2) ** 2

        # initial guess for fit
        p0 = [r1[index], 1., width]
        # x axis
        x = np.arange(peak.size)
        # fit
        coeff0, var_matrix = curve_fit(poly2, x, peak, p0=p0)

        value = coeff0[-1] - width + index - WIDTH_TO_SEARCH
        return value, weight
