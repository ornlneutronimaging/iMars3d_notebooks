import matplotlib.pyplot as plt
import timeit
import numpy as np
import os

from imars3d.backend.preparation.normalization import normalization
from imars3d.backend.dataio.data import save_data

from __code.parent import Parent
from __code.file_folder_browser import FileFolderBrowser


class Normalization(Parent):

    def normalization_and_display(self):
        print(f"Running normalization ...")
        t0 = timeit.default_timer()
        self.parent.proj_norm = normalization(arrays=self.parent.proj_gamma,
                                              flats=self.parent.ob_crop,
                                              darks=self.parent.dc_crop)
        del self.parent.proj_gamma
        t1 = timeit.default_timer()
        print(f"normalization done in {t1 - t0:.2f}s")

        plt.figure()
        self.parent.proj_norm_min = np.min(self.parent.proj_norm, axis=0)
        plt.imshow(self.parent.proj_norm_min)
        plt.colorbar()

    def export_normalization(self):
        working_dir = os.path.join(self.parent.working_dir, "shared", "processed_data")
        if not os.path.exists(working_dir):
            working_dir = self.parent.working_dir

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.export_normalized_data)
        list_folder_selected = o_file_browser.select_output_folder(instruction="Select output folder")

    def export_normalized_data(self, folder):
        print(f"New folder will be created in {folder} and called {self.parent.input_folder_base_name}_YYYYMMDDHHMM")
        save_data(data=np.asarray(self.parent.proj_norm),
                  outputbase=folder,
                  name=self.parent.input_folder_base_name + "_normalized")
