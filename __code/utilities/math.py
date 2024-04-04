import numpy as np


def convert_deg_in_rad(array_of_angles_in_deg):
    array_of_angles_in_rad = []
    for angle in array_of_angles_in_deg:
        angle_rad = np.deg2rad(angle)
        array_of_angles_in_rad.append(angle_rad)
    return array_of_angles_in_rad
