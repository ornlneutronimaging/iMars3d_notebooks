import numpy as np

from __code.parent import Parent


class Sinogram(Parent):

    @staticmethod
    def create_sinogram(data_3d=None):
        return np.moveaxis(data_3d, 1, 0)
