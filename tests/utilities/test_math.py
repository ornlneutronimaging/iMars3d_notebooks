from unittest import TestCase
import numpy as np

from __code.utilities import math


class TestMath(TestCase):

    def test_deg_to_rad(self):
        array_of_deg_angle = [0, 90, 180, 270, 360]
        array_of_rad_angle_returned = math.convert_deg_in_rad(array_of_deg_angle)

        array_of_rad_angle_expected = [0, np.pi/2, np.pi, np.pi + np.pi/2, 2*np.pi]
        for _expected, _returned in zip(array_of_rad_angle_expected, array_of_rad_angle_returned):
            assert _expected == _returned
