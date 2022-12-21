import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import timeit
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML

from imars3d.backend.dataio.data import load_data, _get_filelist_by_dir
from imars3d.backend.morph.crop import crop
from imars3d.backend.corrections.gamma_filter import gamma_filter
from imars3d.backend.preparation.normalization import normalization
from imars3d.backend.diagnostics import tilt
# from imars3d.backend.dataio.data import _get_filelist_by_dir
from imars3d.backend.diagnostics.rotation import find_rotation_center
from imars3d.backend.corrections.ring_removal import remove_ring_artifact
from imars3d.backend.reconstruction import recon
from imars3d.backend.dataio.data import save_data
from imars3d.backend.corrections.intensity_fluctuation_correction import normalize_roi

from __code.tilt.direct_minimization import DirectMinimization
from __code.tilt.phase_correlation import PhaseCorrelation
from __code.tilt.use_center import UseCenter

import tomopy

from __code.file_folder_browser import FileFolderBrowser
from __code import DEFAULT_CROP_ROI, NCORE, DEFAULT_BACKROUND_ROI

IN_PROGRESS = "Calculation in progress"
DONE = "Done!"
QUEUE = "In queue"

class DataType:
    raw = 'raw'
    ob = 'ob'
    dc = 'dc'


default_input_folder = {DataType.raw: 'ct_scans',
                        DataType.ob: 'ob',
                        DataType.dc: 'dc'}


class Imars3dui:

    input_data_folders = {}
    input_files = {}
    crop_roi = DEFAULT_CROP_ROI
    background_roi = DEFAULT_BACKROUND_ROI

    # name of raw folder (used to customize output file name)
    input_folder_base_name = None

    # data arrays
    proj_raw = None
    ob_raw = None
    dc_raw = None

    def __init__(self, working_dir="./"):
        self.working_dir = working_dir

    def select_raw(self):
        self.select_folder(data_type=DataType.raw)

    def select_ob(self):
        self.select_folder(data_type=DataType.ob,
                           multiple_flag=True)

    def select_dc(self):
        self.select_folder(data_type=DataType.dc,
                           multiple_flag=True)

    def select_folder(self, data_type=DataType.raw, multiple_flag=False):
        self.current_data_type = data_type
        working_dir = os.path.join(self.working_dir, 'raw', default_input_folder[data_type])

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.data_selected)
        list_folder_selected = o_file_browser.select_input_folder(instruction=
                                                                  f"Select Folder of {data_type}",
                                                                  multiple_flag=multiple_flag)

    def data_selected(self, list_folders):
        self.input_data_folders[self.current_data_type] = list_folders

        if self.current_data_type == DataType.raw:
            list_folders = [os.path.abspath(list_folders)]
        else:
            list_folders = [os.path.abspath(_folder) for _folder in list_folders]

        list_files = Imars3dui.retrieve_list_of_files(list_folders)
        self.input_files[self.current_data_type] = list_files

        if self.current_data_type == DataType.raw:
            self.input_folder_base_name = os.path.basename(list_folders[0])

        print(f"{self.current_data_type} folder selected: {list_folders} with {len(list_files)} files)")

    @staticmethod
    def retrieve_list_of_files(list_folders):
        list_files = []
        for _folder in list_folders:
            _tiff_files = glob.glob(os.path.join(_folder, "*.tif*"))
            list_files = [*list_files, *_tiff_files]

        list_files.sort()
        return list_files

    def saving_list_of_files(self):
        raw_folder = self.input_data_folders[DataType.raw]
        self.input_files[DataType.raw] = self.retrieve_list_of_files(raw_folder)
        ob_folder = self.input_data_folders[DataType.ob]
        self.input_files[DataType.ob] = self.retrieve_list_of_files(ob_folder)
        dc_folder = self.input_data_folders[DataType.dc]
        self.input_files[DataType.dc] = self.retrieve_list_of_files(dc_folder)

    def load_and_display_data(self):
        self.proj_raw, self.ob_raw, self.dc_raw, self.rot_angles = load_data(ct_files=self.input_files[DataType.raw],
                                                                             ob_files=self.input_files[DataType.ob],
                                                                             dc_files=self.input_files[DataType.dc])

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(5, 9))
        proj_min = np.min(self.proj_raw, axis=0)
        self.proj_min = proj_min
        ob_max = np.max(self.ob_raw, axis=0)
        dc_max = np.max(self.dc_raw, axis=0)

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

    def crop_and_display_data(self, crop_region):
        print(f"Running crop ...")
        self.proj_crop = crop(arrays=self.proj_raw,
                              crop_limit=crop_region)
        self.ob_crop = crop(arrays=self.ob_raw,
                            crop_limit=crop_region)
        self.dc_crop = crop(arrays=self.dc_raw,
                            crop_limit=crop_region)

        self.proj_crop_min = crop(arrays=self.proj_min,
                                  crop_limit=crop_region)
        print(f"cropping done!")

    def gamma_filtering(self):
        print(f"Running gamma filtering ...")
        t0 = timeit.default_timer()
        self.proj_gamma = gamma_filter(arrays=self.proj_crop.astype(np.uint16),
                                       selective_median_filter=False,
                                       diff_tomopy=20,
                                       max_workers=NCORE,
                                       median_kernel=3)
        del self.proj_crop
        t1 = timeit.default_timer()
        print(f"Gamma filtering done in {t1-t0:.2f}s")

    def normalization_and_display(self):
        print(f"Running normalization ...")
        t0 = timeit.default_timer()
        self.proj_norm = normalization(arrays=self.proj_gamma,
                                       flats=self.ob_crop,
                                       darks=self.dc_crop)
        del self.proj_gamma
        t1 = timeit.default_timer()
        print(f"normalization done in {t1-t0:.2f}s")

        plt.figure()
        proj_norm_min = np.min(self.proj_norm, axis=0)
        plt.imshow(proj_norm_min)
        plt.colorbar()

    def beam_fluctuation_correction(self, background_region):
        # [top, left, bottom, right]
        roi = [background_region[2], background_region[0],
               background_region[3], background_region[1]]

        # self.proj_norm_beam_fluctuation = tomopy.prep.normalize.normalize_roi(
        #     self.proj_norm,
        #     roi=roi,
        #     ncore=NCORE)

        self.proj_norm_beam_fluctuation = normalize_roi(
            ct=self.proj_norm,
            roi=roi,
            max_workers=NCORE)

        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,
                                       num="Beam fluctuation",
                                       figsize=(5,10))
        # before beam fluctuation
        #proj_norm_min = np.min(proj_norm, axis=0)
        #fig0 = ax0.imshow(proj_norm_min)
        fig0 = ax0.imshow(self.proj_norm[0])
        ax0.set_title("before")
        plt.colorbar(fig0, ax=ax0)

        # after beam fluctuation
        # proj_norm_beam_fluctuation_min = np.min(proj_norm_beam_fluctuation, axis=0)
        # fig1 = ax1.imshow(proj_norm_beam_fluctuation_min)
        fig1 = ax1.imshow(self.proj_norm_beam_fluctuation[0])
        ax1.set_title("after")
        plt.colorbar(fig1, ax=ax1)

    def minus_log_and_display(self):
        del self.proj_norm
        self.proj_mlog = tomopy.minus_log(self.proj_norm_beam_fluctuation)
        del self.proj_norm_beam_fluctuation
        plt.figure()
        plt.imshow(self.proj_mlog[0])
        plt.colorbar()

    def find_0_180_degrees_files(self):
        rot_angles_sorted = self.rot_angles
        rot_angles_sorted.sort()

        self.mean_delta_angle = np.mean([y - x for (x, y) in zip(rot_angles_sorted[:-1],
                                                            rot_angles_sorted[1:])])

        list_180_deg_pairs_idx = tilt.find_180_deg_pairs_idx(angles=self.rot_angles,
                                                             atol=self.mean_delta_angle)

        index_0_degree = list_180_deg_pairs_idx[0][0]
        index_180_degree = list_180_deg_pairs_idx[1][0]

        list_ct_files = self.input_files[DataType.raw]
        short_list_cf_files = [os.path.basename(_file) for _file in list_ct_files]

        #left panel
        left_label = widgets.Label("0 degree file")
        self.left_select = widgets.Select(options=short_list_cf_files,
                                     value=short_list_cf_files[index_0_degree],
                                     layout=widgets.Layout(width="500px",
                                                           height="400px"))
        left_vbox = widgets.VBox([left_label, self.left_select])

        right_label = widgets.Label("180 degree file")
        self.right_select = widgets.Select(options=short_list_cf_files,
                                     value=short_list_cf_files[index_180_degree],
                                     layout=widgets.Layout(width="500px",
                                                           height="400px"))
        right_vbox = widgets.VBox([right_label, self.right_select])

        hbox = widgets.HBox([left_vbox, right_vbox])
        display(hbox)

    # tilt correction
    def tilt_checkbox_handler(self, checkbox_index=1):

        list_checkboxes = [self.tilt_checkbox1,
                           self.tilt_checkbox2,
                           self.tilt_checkbox3,
                           self.tilt_checkbox4]
        full_list_checkboxes = list_checkboxes[:]
        list_methods = [self.tilt_checkbox1_changed,
                        self.tilt_checkbox2_changed,
                        self.tilt_checkbox3_changed,
                        self.tilt_checkbox4_changed]

        input_checkbox = list_checkboxes.pop(checkbox_index - 1)
        other_checkboxes = list_checkboxes

        for _check, _method in zip(full_list_checkboxes, list_methods):
            _check.unobserve(_method, names='value')

        new_value = input_checkbox.value
        if new_value is False:
            input_checkbox.value = True
            for _check, _method in zip(full_list_checkboxes, list_methods):
                _check.observe(_method, names='value')
                return

        for _check in other_checkboxes:
            _check.value = not new_value

        for _check, _method in zip(full_list_checkboxes, list_methods):
            _check.observe(_method, names='value')

    def tilt_checkbox1_changed(self, value):
        self.tilt_checkbox_handler(checkbox_index=1)

    def tilt_checkbox2_changed(self, value):
        self.tilt_checkbox_handler(checkbox_index=2)

    def tilt_checkbox3_changed(self, value):
        self.tilt_checkbox_handler(checkbox_index=3)

    def tilt_checkbox4_changed(self, value):
        self.tilt_checkbox_handler(checkbox_index=4)

    def calculate_tilt(self):
        # calculate the tilt using all 3 methods and let the user chose the one he wants to apply on the data
        display(HTML('<span style="font-size: 15px; color:blue">Select the tilt value you want to use:</span>'))

        line1 = widgets.HBox([widgets.Checkbox(value=True,
                                               description="Direct minimization"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label(IN_PROGRESS,
                                            layout=widgets.Layout(width="200px"))
                              ])
        self.tilt_checkbox1 = line1.children[0]
        self.direct_minimization_value = line1.children[1]
        direct_minimization_status = line1.children[3]

        line2 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="Phase correlation"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label(QUEUE,
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.tilt_checkbox2 = line2.children[0]
        self.phase_correlation_value = line2.children[1]
        phase_correlation_status = line2.children[3]

        line3 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="Use center"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label(QUEUE,
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.tilt_checkbox3 = line3.children[0]
        self.use_center_value = line3.children[1]
        use_center_status = line3.children[3]

        line4 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="User defined"),
                              widgets.FloatText(0,

                                                layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label("",
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.tilt_checkbox4 = line4.children[0]
        self.user_value = line4.children[1]

        self.tilt_checkbox1.observe(self.tilt_checkbox1_changed, names="value")
        self.tilt_checkbox2.observe(self.tilt_checkbox2_changed, names="value")
        self.tilt_checkbox3.observe(self.tilt_checkbox3_changed, names="value")
        self.tilt_checkbox4.observe(self.tilt_checkbox4_changed, names="value")

        vertical_layout = widgets.VBox([line1, line2, line3, line4])
        display(vertical_layout)

        # direct minimization
        # o_direct = DirectMinimization(index_0_degree=self.index_0_degree,
        #                               index_180_degree=self.index_180_degree)
        # tilt_value = o_direct.compute()
        # self.direct_minimization_value.value = f"{tilt_value:.3f}"
        direct_minimization_status.value = DONE
        phase_correlation_status.value = IN_PROGRESS

        # phase correlation

        phase_correlation_status.value = DONE
        use_center_status.value = IN_PROGRESS

        # use center


        use_center_status.value = DONE







        # print("Calculating tilt ...")
        # self.index_0_degree = self.left_select.index
        # self.index_180_degree = self.right_select.index
        #
        # tilt_angle = tilt.calculate_tilt(image0=self.proj_mlog[self.index_0_degree],
        #                                  image180=self.proj_mlog[self.index_180_degree])
        # # self.tilt_angle = tilt_angle.x
        #
        # label = widgets.Label("Tilt value (degrees):",
        #                       layout=widgets.Layout(width="130px"))
        # self.tilt = widgets.BoundedFloatText(value=tilt_angle.x,
        #                                      min=-5.,
        #                                      max=5.,
        #                                      step=0.01,
        #                                      )
        # hbox = widgets.HBox([label, self.tilt])
        # display(hbox)

    def apply_tilt_and_display(self):
        print("Applying tilt correction ...")
        self.proj_tilt_corrected = tilt.apply_tilt_correction(arrays=self.proj_mlog,
                                                              tilt=self.tilt.value)

        fig, ax = plt.subplots(nrows=1, ncols=1, num="Tilt Correction", figsize=(10, 10))

        index_0_image = self.proj_tilt_corrected[self.index_0_degree]
        index_180_image_flipped = np.fliplr(self.proj_tilt_corrected[self.index_180_degree])
        overlap_image = np.add(index_0_image, index_180_image_flipped)/2.
        fig0 = ax.imshow(overlap_image)
        plt.colorbar(fig0, ax=ax)

    # def tilt_correction_and_display(self):
    #     print("Applying tilt correction ...")
    #     self.proj_tilt_corrected = tilt.apply_tilt_correction(arrays=self.proj_mlog,
    #                                                     tilt=self.tilt_angle)
    #     del self.proj_mlog
    #     print(" tilt correction done!")
    #
    #     fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,
    #                                    num="Tilt correction",
    #                                    figsize=(5, 10))
    #
    #     # before beam fluctuation
    #     #proj_norm_min = np.min(proj_norm, axis=0)
    #     #fig0 = ax0.imshow(proj_norm_min)
    #     fig0 = ax0.imshow(self.proj_tilt_corrected[index_0_degree])
    #     ax0.set_title("0 degree")
    #     plt.colorbar(fig0, ax=ax0)
    #
    #     # after beam fluctuation
    #     # proj_norm_beam_fluctuation_min = np.min(proj_norm_beam_fluctuation, axis=0)
    #     # fig1 = ax1.imshow(proj_norm_beam_fluctuation_min)
    #     fig1 = ax1.imshow(np.fliplr(self.proj_tilt_corrected[index_180_degree]))
    #     ax1.set_title("180 degree (flipped)")
    #     plt.colorbar(fig1, ax=ax1)

    def strikes_removal(self):
        t0 = timeit.default_timer()
        print("Running strikes removal ...")
        self.proj_strikes_removed = remove_ring_artifact(arrays=self.proj_tilt_corrected,
                                                         kernel_size=5,
                                                         max_workers=NCORE)
        print(" strikes removal done!")
        t1 = timeit.default_timer()
        print(f"time= {t1 - t0:.2f}s")

    def rotation_center(self):
        print(f"Running rotation center ...")
        t0 = timeit.default_timer()
        self.rot_center = find_rotation_center(arrays=self.proj_tilt_corrected,
                                               angles=self.rot_angles,
                                               num_pairs=-1,
                                               in_degrees=True,
                                               atol_deg=self.mean_delta_angle,
                                               )
        t1 = timeit.default_timer()
        print(f"rotation center found in {t1-t0:.2f}s")
        print(f" - value: {self.rot_center}")

    def reconstruction_and_display(self):
        t0 = timeit.default_timer()
        print("Running reconstruction ...")

        # converting angles from deg to radians
        rot_ang_rad = np.radians(self.rot_angles)

        self.reconstruction = recon(arrays=self.proj_strikes_removed,
                                    center=self.rot_center[0],
                                    theta=rot_ang_rad,
                                    )

        print(" reconstruction done!")
        t1 = timeit.default_timer()
        print(f"time= {t1 - t0:.2f}s")

        plt.figure()
        plt.imshow(self.reconstruction[0])
        plt.colorbar()
        plt.show()

        def plot_reconstruction(index):
            plt.title(f"Reconstructed slice #{index}")
            plt.imshow(self.reconstruction[index])
            plt.show()

        plot_reconstruction_ui = interactive(plot_reconstruction,
                                             index=widgets.IntSlider(min=0,
                                                                     max=len(self.reconstruction),
                                                                     value=0))
        display(plot_reconstruction_ui)

    def export(self):
        working_dir = os.path.join(self.working_dir, "shared", "processed_data")
        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.export_data)
        list_folder_selected = o_file_browser.select_output_folder(instruction="Select output folder")

    def export_data(self, folder):
        print(f"New folder will be created in {folder} and called {self.input_folder_base_name}_YYYYMMDDHHMM")
        save_data(data=np.asarray(self.reconstruction),
                  outputbase=folder,
                  name=self.input_folder_base_name)
