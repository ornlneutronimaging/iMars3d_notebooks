import tomopy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from __code.parent import Parent
from __code import NCORE, STEP_SIZE
from __code.utilities.system import print_memory_usage, delete_array


class TransmissionToAttenuation(Parent):

    def minus_log_and_display(self):

        print_memory_usage("Before transmission to attenuation")

        # inplace operation to reduce memory pressure
        num_proj = self.parent.proj_norm_beam_fluctuation.shape[0]
        step_size = NCORE * STEP_SIZE
        for i in tqdm(range(0, num_proj, step_size)):
            end_idx = min(i + step_size, num_proj)
            self.parent.proj_norm_beam_fluctuation[i:end_idx] = tomopy.minus_log(
                self.parent.proj_norm_beam_fluctuation[i:end_idx]
            )
        # rename array
        self.parent.proj_mlog = self.parent.proj_norm_beam_fluctuation
        delete_array(self.parent.proj_norm_beam_fluctuation)

        # visualization
        plt.figure()
        plt.imshow(self.parent.proj_mlog[0])
        plt.colorbar()

        print_memory_usage("After transmission to attenuation")
