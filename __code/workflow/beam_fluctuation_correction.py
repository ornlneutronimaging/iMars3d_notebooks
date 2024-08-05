import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
from tqdm.auto import tqdm

from imars3d.backend.corrections.intensity_fluctuation_correction import normalize_roi

from __code.parent import Parent
from __code import NCORE, STEP_SIZE
from __code.utilities.system import print_memory_usage, delete_array


class BeamFluctuationCorrection(Parent):

    def beam_fluctuation_correction_option(self):
        self.parent.beam_fluctuation_ui = widgets.Checkbox(value=False,
                                                           description="Beam fluctuation correction")
        display(self.parent.beam_fluctuation_ui)

    def select_beam_fluctuation_region(self, batch_mode=False):

        if batch_mode:
            display_mode = True
        else:
            display_mode = self.parent.beam_fluctuation_ui.value

        if display_mode:

            if batch_mode:
                integrated_image = self.parent.integrated_proj_min
                
            else:
                proj_norm = self.parent.proj_norm
                integrated_image = np.mean(proj_norm, axis=0)
    
            height, width = np.shape(integrated_image)

            left = self.parent.background_roi[0] if self.parent.background_roi[0] else 0
            right = self.parent.background_roi[1] if self.parent.background_roi[1] else width - 1
            top = self.parent.background_roi[2] if self.parent.background_roi[2] else 0
            bottom = self.parent.background_roi[3] if self.parent.background_roi[3] else height - 1
            vmin = np.nanmin(integrated_image)
            vmax = np.nanmax(integrated_image)

            def plot_crop(left, right, top, bottom, vmin, vmax):

                _, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
                ax.imshow(integrated_image, vmin=vmin, vmax=vmax)

                ax.axvline(left, color='blue', linestyle='--')
                ax.axvline(right, color='red', linestyle='--')

                ax.axhline(top, color='blue', linestyle='--')
                ax.axhline(bottom, color='red', linestyle='--')

                return left, right, top, bottom

            self.parent.beam_fluctuation_roi = interactive(plot_crop,
                                                    left=widgets.IntSlider(min=0,
                                                                           max=width - 1,
                                                                           value=left,
                                                                           continuous_update=True),
                                                    right=widgets.IntSlider(min=0,
                                                                            max=width - 1,
                                                                            value=right,
                                                                            continuous_update=False),
                                                    top=widgets.IntSlider(min=0,
                                                                          max=height - 1,
                                                                          value=top,
                                                                          continuous_update=False),
                                                    bottom=widgets.IntSlider(min=0,
                                                                             max=height - 1,
                                                                             value=bottom,
                                                                             continuous_update=False),
                                                    vmin=widgets.IntSlider(min=0,
                                                                           max=vmax,
                                                                           value=0,
                                                                           continuous_update=False),
                                                    vmax=widgets.IntSlider(min=0,
                                                                           max=vmax,
                                                                           value=vmax,
                                                                           continuous_update=False),
                                                                    )
            display(self.parent.beam_fluctuation_roi)

        else:
            self.parent.proj_norm_beam_fluctuation = self.parent.proj_norm

    def beam_fluctuation_correction_embedded(self):
        if self.parent.beam_fluctuation_ui.value:
            background_region = list(self.parent.beam_fluctuation_roi.result)
            self._beam_fluctuation(background_region=background_region)

    def _beam_fluctuation(self, background_region=None):

        print_memory_usage("Before beam fluctuation")

        # build roi
        if background_region is None:
            print("No background region selected, skip roi beam fluctuation correction")
            return

        roi = [
            background_region[2],  # top
            background_region[0],  # left
            background_region[3],  # bottom
            background_region[1],  # right
        ]

        # cache the before image for display later
        proj_before = np.array(self.parent.proj_norm[0])

        num_proj = self.parent.proj_norm.shape[0]
        step_size = NCORE * STEP_SIZE
        for i in tqdm(range(0, num_proj, step_size)):
            end_idx = min(i + step_size, num_proj)
            self.parent.proj_norm_beam_fluctuation[i: end_idx] = normalize_roi(
                    ct=self.parent.proj_norm[i: end_idx],
                    roi=roi,
                    max_workers=NCORE,
        )

        delete_array(self.parent.proj_norm)

        _, (ax0, ax1) = plt.subplots(
            nrows=2,
            ncols=1,
            num="Beam fluctuation",
            figsize=(5, 10),
        )
        # before beam fluctuation
        fig0 = ax0.imshow(proj_before)
        ax0.set_title("before")
        plt.colorbar(fig0, ax=ax0)

        # after beam fluctuation
        fig1 = ax1.imshow(self.parent.proj_norm_beam_fluctuation[0])
        ax1.set_title("after")
        plt.colorbar(fig1, ax=ax1)

        print_memory_usage("After beam fluctuation")
