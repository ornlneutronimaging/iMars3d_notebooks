import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display

from __code.parent import Parent


class SelectZRange(Parent):

    def select_range_of_slices(self):
        list_images = self.parent.proj_tilt_corrected
        integrated_image = np.mean(list_images, axis=0)
        height, width = np.shape(integrated_image)

        z_top = self.parent.crop_roi[2] if self.parent.crop_roi[2] else 0
        z_bottom = self.parent.crop_roi[3] if self.parent.crop_roi[3] else height - 1

        def plot_z_range(top, bottom):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            ax.imshow(integrated_image)

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
                                                )
        display(self.parent.z_range_selection)
