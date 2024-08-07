import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display

from __code.parent import Parent


class SelectZRange(Parent):

    def select_range_of_slices(self, batch_mode=False):

        if batch_mode:
            integrated_image = self.parent.integrated_proj_min
        else:
            list_images = self.parent.proj_tilt_corrected
            integrated_image = np.mean(list_images, axis=0)
        height, width = np.shape(integrated_image)
        max_value = np.max(integrated_image)

        z_top = self.parent.crop_roi[2] if self.parent.crop_roi[2] else 0
        z_bottom = self.parent.crop_roi[3] if self.parent.crop_roi[3] else height - 1

        def plot_z_range(top, bottom, vmin, vmax):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            ax.imshow(integrated_image, vmin=vmin, vmax=vmax)

            ax.axhline(top, color='blue', linestyle='--')
            ax.axhline(bottom, color='red', linestyle='--')

            return top, bottom

        self.parent.z_range_selection = interactive(plot_z_range,
                                                top=widgets.IntSlider(min=0,
                                                                    max=height - 1,
                                                                    value=z_top,
                                                                    continuous_update=False),
                                                bottom=widgets.IntSlider(min=0,
                                                                        max=height - 1,
                                                                        value=z_bottom,
                                                                        continuous_update=False),
                                                vmin=widgets.IntSlider(min=0,
                                                                   max=max_value,
                                                                   value=0,
                                                                   continuous_update=False),
                                            vmax=widgets.IntSlider(min=0,
                                                                   max=max_value,
                                                                   value=max_value,
                                                                   continuous_update=False)
                                                )
        display(self.parent.z_range_selection)
