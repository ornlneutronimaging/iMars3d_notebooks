import tomopy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from __code.parent import Parent
from __code import NCORE

class TransmissionToAttenuation(Parent):

    def minus_log_and_display(self):
        # inplace operation to reduce memory pressure
        num_proj = self.parent.proj_norm_beam_fluctuation.shape[0]
        step_size = NCORE * 5
        for i in tqdm(range(0, num_proj, step_size)):
            end_idx = min(i + step_size, num_proj)
            self.parent.proj_norm_beam_fluctuation[i:end_idx] = tomopy.minus_log(
                self.parent.proj_norm_beam_fluctuation[i:end_idx]
            )
        # rename array
        self.parent.proj_mlog = self.parent.proj_norm_beam_fluctuation
        # cleanup (just reduce counter here)
        print("Deleting proj_norm_beam_fluctuation and releasing memory ...")
        self.parent.proj_norm_beam_fluctuation = None
        import gc
        gc.collect()
        # visualization
        plt.figure()
        plt.imshow(self.parent.proj_mlog[0])
        plt.colorbar()
