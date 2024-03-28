from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
import timeit
import numpy as np
from __code import NCORE

from imars3d.backend.corrections.gamma_filter import gamma_filter

from __code.parent import Parent


class GammaFiltering(Parent):

    def gamma_filtering_options(self):

        self.parent.gamma_filtering_ui = widgets.Checkbox(value=False,
                                                          description="Gamma filtering")
        display(self.parent.gamma_filtering_ui)

    def gamma_filtering(self):
        if self.parent.gamma_filtering_ui.value:
            print(f"Running gamma filtering ...")
            t0 = timeit.default_timer()
            self.parent.proj_gamma = gamma_filter(arrays=self.parent.proj_crop.astype(np.uint16),
                                                  selective_median_filter=False,
                                                  diff_tomopy=20,
                                                  max_workers=NCORE,
                                                  median_kernel=3)
            del self.parent.proj_crop
            t1 = timeit.default_timer()
            print(f"Gamma filtering done in {t1 - t0:.2f}s")
        else:
            self.parent.proj_gamma = self.parent.proj_crop
            print("Gamma filtering skipped!")
