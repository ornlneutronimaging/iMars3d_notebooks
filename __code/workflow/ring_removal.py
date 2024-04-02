from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
import timeit
import tomopy
import numpy as np
import matplotlib.pyplot as plt

try:
    from imars3d.backend.corrections.ring_removal import remove_ring_artifact
    enable_remove_ring_artifact = True
except OSError:
    enable_remove_ring_artifact = False

from __code.parent import Parent
from __code import NCORE


class RingRemoval(Parent):

    def ring_removal_options(self):
        self.ring_removal_ui = widgets.VBox([widgets.Checkbox(value=False,
                                                              description="BM3D",
                                                              disabled=True),
                                             widgets.Checkbox(value=False,
                                                              description="Tomopy (Vo)"),
                                             widgets.Checkbox(value=False,
                                                              description="Ketcham")],
                                            layout={'width': 'max-content'},
                                            )
        display(self.ring_removal_ui)

    def apply_ring_removal_options(self):

        # bm3d
        if self.ring_removal_ui.children[0].value:
            t0 = timeit.default_timer()
            print("Running strikes removal using BM3D ...")
            import bm3d_streak_removal as bm3d
            proj_mlog_bm3d = bm3d.extreme_streak_attenuation(self.parent.proj_tilt_corrected)
            self.parent.proj_ring_removal_1 = bm3d.multiscale_streak_removal(proj_mlog_bm3d)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using BM3D")
            self.parent.proj_ring_removal_1 = self.parent.proj_tilt_corrected

        # tomopy, Vo
        if self.ring_removal_ui.children[1].value:
            t0 = timeit.default_timer()
            print("Running strikes removal using Vo ...")
            self.parent.proj_ring_removal_2 = tomopy.remove_all_stripe(self.parent.proj_ring_removal_1,
                                                                ncore=NCORE)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using Vo")
            self.parent.proj_ring_removal_2 = self.proj_ring_removal_1

        # ketcham
        if self.ring_removal_ui.children[2].value:
            t0 = timeit.default_timer()
            print("Running strikes removal using Ketcham ...")
            self.parent.proj_strikes_removed = remove_ring_artifact(arrays=self.parent.proj_ring_removal_2,
                                                             kernel_size=5,
                                                             max_workers=NCORE)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using Ketcham")
            self.parent.proj_ring_removal_3 = self.parent.proj_ring_removal_2

    def test_ring_removal(self):

        after_sinogram_mlog = self.parent.proj_ring_removal_3.astype(np.float32)
        after_sinogram_mlog = np.moveaxis(after_sinogram_mlog, 1, 0)

        def plot_test_ring_removal(index):
            fig, axis = plt.subplots(num="sinogram", figsize=(15, 10), nrows=1, ncols=3)

            axis[0].imshow(self.parent.sinogram_mlog[index])
            axis[0].set_title(f"Before ring removal")

            axis[1].imshow(after_sinogram_mlog[index])
            axis[1].set_title(f"After ring removal")

            axis[2].imshow(after_sinogram_mlog[index] - self.parent.sinogram_mlog[index])
            axis[2].set_title(f"Difference")

        plot_test_ui = interactive(plot_test_ring_removal,
                                   index=widgets.IntSlider(min=0,
                                                           max=len(self.parent.sinogram_mlog)))
        display(plot_test_ui)
