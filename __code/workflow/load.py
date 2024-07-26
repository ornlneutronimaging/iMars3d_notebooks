import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
import random

from imars3d.backend.dataio.data import load_data

from __code import DataType
from __code.parent import Parent
from __code.file_folder_browser import FileFolderBrowser
from __code.utilities.files import retrieve_list_of_files


class Load(Parent):

    def select_folder(self, data_type=DataType.raw, multiple_flag=False):
        self.parent.current_data_type = data_type

        if data_type == DataType.dc:

            if not self.parent.select_dc_flag.value:
                print(f"Not using any DC!")
                self.parent.input_files[data_type] = [self.parent.input_files[DataType.ob][0]]  # using dumb file to make iMars3D happy
                return

        working_dir = self.parent.working_dir[data_type]
        if not os.path.exists(working_dir):
            working_dir = os.path.abspath(os.path.expanduser("~"))

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
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

        print(f"{list_folders =}")

        list_files = retrieve_list_of_files(list_folders)
        self.parent.input_files[self.parent.current_data_type] = list_files

        if self.parent.current_data_type == DataType.raw:
            self.parent.input_folder_base_name = os.path.basename(list_folders[0])

        print(f"{self.parent.current_data_type} folder selected: {list_folders} with {len(list_files)} files)")

    def load_and_display_data(self):
        self.load_data()
        self.display_data()

    def load_percentage_of_data(self, percentage_to_load=5):
        nbr_sample_file = len(self.parent.input_files[self.parent.current_data_type])
        nbr_file_to_load = int(percentage_to_load * nbr_sample_file / 100)
        list_file_index_to_use = random.sample(range(1, nbr_sample_file), nbr_file_to_load)
        list_raw_file = []
        for _index in list_file_index_to_use:
            list_raw_file.append(self.parent.input_files[DataType.raw][_index])

        self.parent.proj_raw, self.parent.ob_raw, self.parent.dc_raw, self.parent.rot_angles = (
            load_data(ct_files=list_raw_file,
                      ob_files=self.parent.input_files[DataType.ob],
                      dc_files=self.parent.input_files[DataType.dc],
                      max_workers=20)  # use 20 workers
        )








    def load_data(self):

        self.parent.proj_raw, self.parent.ob_raw, self.parent.dc_raw, self.parent.rot_angles = (
            load_data(ct_files=self.parent.input_files[DataType.raw],
                      ob_files=self.parent.input_files[DataType.ob],
                      dc_files=self.parent.input_files[DataType.dc],
                      max_workers=20)  # use as many CPU as available
        )

        if not self.parent.select_dc_flag.value:
            # create zeros array of dc 

            print(f"{np.shape(self.parent.proj_raw) =}")
            print(f"{np.shape(self.parent.proj_raw[0]) =}")

            self.parent.dc_raw = np.array([np.zeros_like(self.parent.proj_raw[0])])

        self.parent.untouched_sample_data = copy.deepcopy(self.parent.proj_raw)

    def select_dc_options(self):
        self.parent.select_dc_flag = widgets.Checkbox(value=True,
                                                      description="Use dark current")
        display(self.parent.select_dc_flag)

    def display_data(self):

        proj_min = np.min(self.parent.proj_raw, axis=0)
        self.parent.proj_min = proj_min
        ob_max = np.max(self.parent.ob_raw, axis=0)
        self.parent.ob_max = ob_max
        dc_max = np.max(self.parent.dc_raw, axis=0)
        self.parent.dc_max = dc_max
        
        # max_value = np.max(mean_image)

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(5,9))

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

    def investigate_loaded_data_flag(self):
        self.parent.investigate_loaded_data_flag_ui = widgets.Checkbox(value=False,
                                                           description="Investigate data")
        display(self.parent.investigate_loaded_data_flag_ui)

    def investigate_loaded_data(self):
        if self.parent.investigate_loaded_data_flag_ui.value:

            proj_min = self.parent.proj_min

            ob_max = self.parent.ob_max
            dc_max = self.parent.dc_max
            mean_image = np.mean(self.parent.proj_raw, axis=0)
            max_value = np.max(mean_image)

            def plot_data(vmin, vmax):

                fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, figsize=(5,14))
    
                plt0 = ax0.imshow(mean_image, vmin=vmin, vmax=vmax)
                fig.colorbar(plt0, ax=ax0)
                ax0.set_title("np.mean(proj_raw)")

                plt1 = ax1.imshow(proj_min, vmin=vmin, vmax=vmax)
                fig.colorbar(plt1, ax=ax1)
                ax1.set_title("np.min(proj_raw)")

                plt2 = ax2.imshow(ob_max, vmin=vmin, vmax=vmax)
                fig.colorbar(plt2, ax=ax2)
                ax2.set_title("np.max(ob_raw)")

                plt3 = ax3.imshow(dc_max, vmin=vmin, vmax=vmax)
                fig.colorbar(plt3, ax=ax3)
                ax3.set_title("np.max(dc_raw)")

                fig.tight_layout()

            preview_loaded = interactive(plot_data,
                                        vmin=widgets.IntSlider(min=0,
                                                               max=max_value,
                                                               value=0,
                                                               continuous_update=False),
                                        vmax=widgets.IntSlider(min=0,
                                                               max=max_value,
                                                               value=max_value,
                                                               continuous_update=False)
                                                                    
                                        )
            display(preview_loaded)

        else:
            print("No advanced visualization of loaded data requested!")
