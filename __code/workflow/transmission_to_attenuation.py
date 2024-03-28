import tomopy
import matplotlib.pyplot as plt

from __code.parent import Parent

class TransmissionToAttenuation(Parent):

    def minus_log_and_display(self):
        del self.parent.proj_norm
        self.parent.proj_mlog = tomopy.minus_log(self.parent.proj_norm_beam_fluctuation)
        del self.parent.proj_norm_beam_fluctuation
        plt.figure()
        plt.imshow(self.parent.proj_mlog[0])
        plt.colorbar()
