import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display

from imars3d.backend.corrections.intensity_fluctuation_correction import normalize_roi

from __code.parent import Parent
from __code import NCORE


class BeamFluctuationCorrection(Parent):

    def beam_fluctuation_correction_option(self):
        self.parent.beam_fluctuation_ui = widgets.Checkbox(value=False,
                                                           description="Beam fluctuation correction")
        display(self.parent.beam_fluctuation_ui)

    def apply_select_beam_fluctuation(self):

        if self.parent.beam_fluctuation_ui.value:
            integrated_image = np.mean(self.parent.proj_norm, axis=0)
            height, width = np.shape(integrated_image)

            left = self.parent.background_roi[0] if self.parent.background_roi[0] else 0
            right = self.parent.background_roi[1] if self.parent.background_roi[1] else width - 1
            top = self.parent.background_roi[2] if self.parent.background_roi[2] else 0
            bottom = self.parent.background_roi[3] if self.parent.background_roi[3] else height - 1

            def plot_crop(left, right, top, bottom):

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
                ax.imshow(integrated_image)

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
                                                    )
            display(self.parent.beam_fluctuation_roi)

        else:
            self.parent.proj_norm_beam_fluctuation = self.parent.proj_norm

    def beam_fluctuation_correction_embedded(self):
        if self.parent.beam_fluctuation_ui.value:
            background_region = list(self.parent.beam_fluctuation_roi.result)
            self._beam_fluctuation(background_region=background_region)

    def _beam_fluctuation(self, background_region=None):

        # [top, left, bottom, right]
        roi = [background_region[2], background_region[0],
               background_region[3], background_region[1]]

        self.parent.proj_norm_beam_fluctuation = normalize_roi(
                    ct=self.parent.proj_norm,
                    roi=roi,
                    max_workers=NCORE)

        fig, (ax0, ax1) = plt.subplots(nrows=2,
                                       ncols=1,
                                       num="Beam fluctuation",
                                       figsize=(5, 10))
        # before beam fluctuation
        # proj_norm_min = np.min(proj_norm, axis=0)
        # fig0 = ax0.imshow(proj_norm_min)
        fig0 = ax0.imshow(self.parent.proj_norm[0])
        ax0.set_title("before")
        plt.colorbar(fig0, ax=ax0)

        # after beam fluctuation
        # proj_norm_beam_fluctuation_min = np.min(proj_norm_beam_fluctuation, axis=0)
        # fig1 = ax1.imshow(proj_norm_beam_fluctuation_min)
        fig1 = ax1.imshow(self.parent.proj_norm_beam_fluctuation[0])
        ax1.set_title("after")
        plt.colorbar(fig1, ax=ax1)
