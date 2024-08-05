from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
import timeit
import tomopy
import numpy as np
import matplotlib.pyplot as plt
from __code.utilities.system import print_memory_usage, delete_array

try:
    from imars3d.backend.corrections.ring_removal import remove_ring_artifact
    enable_remove_ring_artifact = True
except OSError:
    enable_remove_ring_artifact = False

from __code.parent import Parent
from __code import NCORE


class RingRemoval(Parent):

    def ring_removal_options(self):
        self.parent.ring_removal_ui = widgets.VBox([widgets.Checkbox(value=False,
                                                              description="BM3D",
                                                              disabled=True),
                                             widgets.Checkbox(value=False,
                                                              description="Tomopy (Vo)"),
                                             widgets.Checkbox(value=False,
                                                              description="Ketcham")],
                                            layout={'width': 'max-content'},
                                            )
        display(self.parent.ring_removal_ui)

    def apply_ring_removal_options(self):

        print_memory_usage("Before applying ring removal")

        # bm3d
        if self.parent.ring_removal_ui.children[0].value:
            t0 = timeit.default_timer()
            print("Running strikes removal using BM3D ...")
            import bm3d_streak_removal as bm3d
            proj_mlog_bm3d = bm3d.extreme_streak_attenuation(self.parent.proj_tilt_corrected)
            self.proj_ring_removal_1 = bm3d.multiscale_streak_removal(proj_mlog_bm3d)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using BM3D")
            self.proj_ring_removal_1 = self.parent.proj_tilt_corrected

        # tomopy, Vo
        if self.parent.ring_removal_ui.children[1].value:
            t0 = timeit.default_timer()
            print("Running strikes removal using Vo ...")
            self.proj_ring_removal_2 = tomopy.remove_all_stripe(self.proj_ring_removal_1,
                                                                ncore=NCORE)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using Vo")
            self.proj_ring_removal_2 = self.proj_ring_removal_1

        # ketcham
        if self.parent.ring_removal_ui.children[2].value:
            t0 = timeit.default_timer()
            print("Running strikes removal using Ketcham ...")
            self.proj_ring_removal_3 = remove_ring_artifact(arrays=self.proj_ring_removal_2,
                                                               kernel_size=5,
                                                               max_workers=NCORE)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using Ketcham")
            self.proj_ring_removal_3 = self.proj_ring_removal_2

        self.parent.proj_ring_removed = self.proj_ring_removal_3
        print_memory_usage("After applying ring removal")

    def test_ring_removal(self):

        before_singgram_log = self.parent.sinogram_before_ring_removal

        after_sinogram_mlog = self.parent.proj_ring_removed.astype(np.float32)
        after_sinogram_mlog = np.moveaxis(after_sinogram_mlog, 1, 0)
        self.parent.sinogram_after_ring_removal = after_sinogram_mlog

        def plot_test_ring_removal(index):
            fig, axis = plt.subplots(num="sinogram", figsize=(15, 10), nrows=1, ncols=3)

            axis[0].imshow(before_singgram_log[index])
            axis[0].set_title(f"Before ring removal")

            axis[1].imshow(after_sinogram_mlog[index])
            axis[1].set_title(f"After ring removal")

            axis[2].imshow(after_sinogram_mlog[index] - before_singgram_log[index])
            axis[2].set_title(f"Difference")

        plot_test_ui = interactive(plot_test_ring_removal,
                                   index=widgets.IntSlider(min=0,
                                                           max=len(after_sinogram_mlog)))
        display(plot_test_ui)
