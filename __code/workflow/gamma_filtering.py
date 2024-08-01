import ipywidgets as widgets
from IPython.display import display
import timeit
from tqdm.auto import tqdm
from __code import NCORE

from imars3d.backend.corrections.gamma_filter import gamma_filter

from __code.parent import Parent


class GammaFiltering(Parent):

    def gamma_filtering_options(self):

        self.parent.gamma_filtering_ui = widgets.Checkbox(
            value=False,
            description="Gamma filtering",
        )
        display(self.parent.gamma_filtering_ui)

    def gamma_filtering(self):
        if self.parent.gamma_filtering_ui.value:
            print(f"Running gamma filtering ...")
            t0 = timeit.default_timer()
            # process NCORE * 5 frames at a time
            # inplace operation
            number_of_projections = self.parent.proj_crop.shape[0]
            for i in tqdm(range(0, number_of_projections, NCORE * 5)):
                end_idx = min(i + NCORE * 5, number_of_projections)
                self.parent.proj_crop[i:end_idx] = gamma_filter(
                    arrays=self.parent.proj_crop[i:end_idx],
                    selective_median_filter=False,
                    diff_tomopy=20,
                    max_workers=NCORE,
                    median_kernel=3,
                )
            t1 = timeit.default_timer()
            print(f"Gamma filtering done in {t1 - t0:.2f}s")
        else:
            print("Gamma filtering skipped!")
        # rename array
        self.parent.proj_gamma = self.parent.proj_crop
        # cleanup (just reduce a counter here, not actually releasing anything)
        print("Deleting proj_crop and releasing memory ...")
        self.parent.proj_crop = None
        import gc
        gc.collect()
