import os
import numpy as np
import matplotlib.pyplot as plt

from imars3d.backend.dataio.data import load_data

from __code import DataType
from __code.parent import Parent
from __code.file_folder_browser import FileFolderBrowser
from __code.utilities.files import retrieve_list_of_files

class Load(Parent):

    def select_folder(self, data_type=DataType.raw, multiple_flag=False):
        self.parent.current_data_type = data_type

        if not os.path.exists(self.parent.working_dir):
            self.parent.working_dir = os.path.abspath(os.path.expanduser("~"))

        o_file_browser = FileFolderBrowser(working_dir=self.parent.working_dir,
                                           next_function=self.data_selected)
        o_file_browser.select_input_folder(instruction=f"Select Folder of {data_type}",
                                           multiple_flag=multiple_flag)

    def data_selected(self, list_folders):
        self.parent.input_data_folders[self.parent.current_data_type] = list_folders

        if self.parent.current_data_type == DataType.raw:
            list_folders = [os.path.abspath(list_folders)]
            self.working_dir = os.path.dirname(os.path.dirname(list_folders[0]))  # default folder is the parent folder of sample
        else:
            list_folders = [os.path.abspath(_folder) for _folder in list_folders]

        list_files = retrieve_list_of_files(list_folders)
        self.parent.input_files[self.parent.current_data_type] = list_files

        if self.parent.current_data_type == DataType.raw:
            self.parent.input_folder_base_name = os.path.basename(list_folders[0])

        print(f"{self.parent.current_data_type} folder selected: {list_folders} with {len(list_files)} files)")

    def load_and_display_data(self):
        self.parent.proj_raw, self.parent.ob_raw, self.parent.dc_raw, self.parent.rot_angles = (
            load_data(ct_files=self.parent.input_files[DataType.raw],
                      ob_files=self.parent.input_files[DataType.ob],
                      dc_files=self.parent.input_files[DataType.dc]))

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(5, 9))
        proj_min = np.min(self.parent.proj_raw, axis=0)
        self.parent.proj_min = proj_min
        ob_max = np.max(self.parent.ob_raw, axis=0)
        dc_max = np.max(self.parent.dc_raw, axis=0)

        plt0 = ax0.imshow(proj_min)
        fig.colorbar(plt0, ax=ax0)
        ax0.set_title("np.min(proj_raw)")

        plt1 = ax1.imshow(ob_max)
        fig.colorbar(plt1, ax=ax1)
        ax1.set_title("np.max(ob_raw)")

        plt2 = ax2.imshow(dc_max)
        fig.colorbar(plt2, ax=ax2)
        ax2.set_title("np.max(dc_raw)")

        fig.tight_layout()
