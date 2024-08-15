import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

from imars3d.backend.morph.crop import crop

from __code.utilities.system import delete_array, print_memory_usage
from __code.parent import Parent


class Crop(Parent):

    def crop_embedded(self, batch_mode=False):

        list_images = self.parent.proj_raw
        if batch_mode:
            proj_min = np.mean(list_images, axis=0)
        else:
            proj_min = self.parent.proj_min
    
        self.parent.integrated_proj_min = proj_min
        height, width = np.shape(proj_min)
        max_value = np.max(proj_min)

        crop_left = self.parent.crop_roi[0] if self.parent.crop_roi[0] else 0
        crop_right = self.parent.crop_roi[1] if self.parent.crop_roi[1] else width - 1
        crop_top = self.parent.crop_roi[2] if self.parent.crop_roi[2] else 0
        crop_bottom = self.parent.crop_roi[3] if self.parent.crop_roi[3] else height - 1

        def plot_crop(left, right, top, bottom, vmin, vmax):

            plt.figure(0)
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            ax.imshow(proj_min, vmin=vmin, vmax=vmax)
            ax.set_title("np.mean(proj_raw)")

            ax.axvline(left, color='blue', linestyle='--')
            ax.axvline(right, color='red', linestyle='--')

            ax.axhline(top, color='blue', linestyle='--')
            ax.axhline(bottom, color='red', linestyle='--')

            return left, right+1, top, bottom+1

        self.parent.cropping = interactive(plot_crop,
                                            left=widgets.IntSlider(min=0,
                                                                   max=width - 1,
                                                                   value=crop_left,
                                                                   continuous_update=True),
                                            right=widgets.IntSlider(min=0,
                                                                    max=width - 1,
                                                                    value=crop_right,
                                                                    continuous_update=False),
                                            top=widgets.IntSlider(min=0,
                                                                  max=height - 1,
                                                                  value=crop_top,
                                                                  continuous_update=False),
                                            bottom=widgets.IntSlider(min=0,
                                                                     max=height - 1,
                                                                     value=crop_bottom,
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
        display(self.parent.cropping)

    def crop_region(self, crop_region):
        print_memory_usage(message='Before')
        print(f"Running crop ...")

        print(f"{crop_region =}")
        print(f"{np.shape(self.parent.proj_raw) =}")

        self.parent.proj_crop = crop(arrays=self.parent.proj_raw,
                                     crop_limit=crop_region)
        self.parent.ob_crop = crop(arrays=self.parent.ob_raw,
                                   crop_limit=crop_region)
        self.parent.proj_crop_min = crop(arrays=self.parent.proj_min,
                                         crop_limit=crop_region)

        print(f"cropping done!")

        delete_array(self.parent.proj_raw)
        delete_array(self.parent.ob_raw)
        delete_array(self.parent.proj_min)

        print_memory_usage(message='After')
