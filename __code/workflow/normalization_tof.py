import matplotlib.pyplot as plt
import timeit
import numpy as np
import os
from tqdm.auto import tqdm

from imars3d.backend.preparation.normalization import normalization
from imars3d.backend.dataio.data import save_data

from __code.parent import Parent
from __code.file_folder_browser import FileFolderBrowser
from __code import NCORE, STEP_SIZE
from __code.utilities.system import print_memory_usage, delete_array


class Normalization(Parent):

    def normalization_and_display(self):
        print_memory_usage(message="Before")
        print(f"Running normalization ...")
        t0 = timeit.default_timer()
        
        # note: we need to use in place operation to reduce memory usage
        # step 0: cast to float32 so that we can use proj_gamma as output container
        self.parent.proj_gamma = self.parent.proj_gamma.astype(np.float32)
        # step 1: process NCORE * 5 frames at a time
        num_proj = self.parent.proj_gamma.shape[0]

        step_size = NCORE * STEP_SIZE
        for i in tqdm(range(0, num_proj, step_size)):
            end_idx = min(i + step_size, num_proj)
            self.parent.proj_gamma[i:end_idx] = normalization(
                arrays=self.parent.proj_gamma[i:end_idx],
                flats=self.parent.ob_crop,
                max_workers=NCORE
            )
        # step 2: rename the array
        self.parent.proj_norm = self.parent.proj_gamma
        t1 = timeit.default_timer()
        print(f"normalization done in {t1 - t0:.2f}s")

        # visualization
        plt.figure(num="np.mean(norm)")
        self.parent.proj_norm_min = np.min(self.parent.proj_norm, axis=0)
        proj_norm_for_display = np.mean(self.parent.proj_norm, axis=0)
        plt.imshow(proj_norm_for_display)
        plt.colorbar()

        print_memory_usage(message="After")
        delete_array(self.parent.proj_gamma)

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
