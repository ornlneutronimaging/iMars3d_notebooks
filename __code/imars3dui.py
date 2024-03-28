import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import timeit
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML
import algotom.rec.reconstruction as rec
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

from imars3d.backend.dataio.data import load_data
from imars3d.backend.morph.crop import crop
from imars3d.backend.corrections.gamma_filter import gamma_filter
from imars3d.backend.preparation.normalization import normalization
from imars3d.backend.diagnostics import tilt
# from imars3d.backend.dataio.data import _get_filelist_by_dir
from imars3d.backend.diagnostics.rotation import find_rotation_center

try:
    from imars3d.backend.corrections.ring_removal import remove_ring_artifact
    enable_remove_ring_artifact = True
except OSError:
    enable_remove_ring_artifact = False

from imars3d.backend.reconstruction import recon
from imars3d.backend.dataio.data import save_data
from imars3d.backend.corrections.intensity_fluctuation_correction import normalize_roi

from __code import DataType
from __code.workflow.load import Load
from __code.workflow.crop import Crop
from __code.workflow.gamma_filtering import GammaFiltering

from __code.tilt.direct_minimization import DirectMinimization
from __code.tilt.phase_correlation import PhaseCorrelation
from __code.tilt.use_center import UseCenter
from __code import config

import tomopy

from __code.file_folder_browser import FileFolderBrowser
from __code import NCORE

IN_PROGRESS = "Calculation in progress"
DONE = "Done!"
QUEUE = "In queue"


# class DataType:
#     raw = 'raw'
#     ob = 'ob'
#     dc = 'dc'


class TiltAlgorithms:
    phase_correlation = "phase correlation"
    direct_minimization = "direct minimization"
    use_center = "use center"
    scipy_minimizer = "SciPy minimizer"
    user = "user"


default_input_folder = {DataType.raw: 'ct_scans',
                        DataType.ob: 'ob',
                        DataType.dc: 'dc'}


class Imars3dui:

    input_data_folders = {}
    input_files = {}

    if config.debugging:
        crop_roi = config.DEFAULT_CROP_ROI
        background_roi = config.DEFAULT_BACKROUND_ROI
    else:
        crop_roi = [None, None, None, None]
        background_roi = [None, None, None, None]

    dict_tilt_values = {}

    # name of raw folder (used to customize output file name)
    input_folder_base_name = None

    # data arrays
    proj_raw = None
    ob_raw = None
    dc_raw = None

    def __init__(self, working_dir="./"):
        # working_dir = self.find_first_real_dir(start_dir=working_dir)
        self.working_dir = os.path.join(working_dir, 'raw', default_input_folder[DataType.raw])

    # SELECT INPUT DATA ===============================================================================================
    def select_raw(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.raw)

    def select_ob(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.ob,
                             multiple_flag=True)

    def select_dc(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.dc,
                             multiple_flag=True)

    def data_selected(self, list_folders):
        o_load = Load(parent=self)
        o_load.data_selected(list_folders=list_folders)

    def saving_list_of_files(self):
        raw_folder = self.input_data_folders[DataType.raw]
        self.input_files[DataType.raw] = self.retrieve_list_of_files(raw_folder)
        ob_folder = self.input_data_folders[DataType.ob]
        self.input_files[DataType.ob] = self.retrieve_list_of_files(ob_folder)
        dc_folder = self.input_data_folders[DataType.dc]
        self.input_files[DataType.dc] = self.retrieve_list_of_files(dc_folder)

    def load_and_display_data(self):
        o_load = Load(parent=self)
        o_load.load_and_display_data()

    # CROP ===============================================================================================

    def crop_embedded(self):
        o_crop = Crop(parent=self)
        o_crop.crop_embedded()

    def saving_crop_region(self, crop_region):
        self.crop_region = crop_region

    def perform_embedded_cropping(self):
        crop_region = list(self.cropping.result)
        o_crop = Crop(parent=self)
        o_crop.crop_region(crop_region)

    def perform_cropping(self):
        crop_region = self.crop_region
        self._crop_region(crop_region=crop_region)

    # GAMMA_FILTERING =====================================================================================

    def gamma_filtering_options(self):
        o_gamma = GammaFiltering(parent=self)
        o_gamma.gamma_filtering_options()

    def gamma_filtering(self):
        o_gamma = GammaFiltering(parent=self)
        o_gamma.gamma_filtering()

    # NORMAlIZATION =====================================================================================

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
        self.proj_norm_min = np.min(self.proj_norm, axis=0)
        plt.imshow(self.proj_norm_min)
        plt.colorbar()

    def beam_fluctuation_correction_option(self):
        self.beam_fluctuation_ui = widgets.Checkbox(value=False,
                                                    description="Beam fluctuation correction")
        display(self.beam_fluctuation_ui)

    def apply_select_beam_fluctuation(self):

        if self.beam_fluctuation_ui.value:
            integrated_image = np.mean(self.proj_norm, axis=0)
            height, width = np.shape(integrated_image)

            left = self.background_roi[0] if self.background_roi[0] else 0
            right = self.background_roi[1] if self.background_roi[1] else width-1
            top = self.background_roi[2] if self.background_roi[2] else 0
            bottom = self.background_roi[3] if self.background_roi[3] else height-1

            def plot_crop(left, right, top, bottom):

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
                ax.imshow(integrated_image)

                ax.axvline(left, color='blue', linestyle='--')
                ax.axvline(right, color='red', linestyle='--')

                ax.axhline(top, color='blue', linestyle='--')
                ax.axhline(bottom, color='red', linestyle='--')

                return left, right, top, bottom

            self.beam_fluctuation_roi = interactive(plot_crop,
                                        left=widgets.IntSlider(min=0,
                                                               max=width - 1,
                                                               value=left,
                                                               continuous_update=True),
                                        right=widgets.IntSlider(min=0,
                                                                max=width - 1,
                                                                value=right,
                                                                continuous_update=False),
                                        top=widgets.IntSlider(min=0,
                                                              max=height - 1,
                                                              value=top,
                                                              continuous_update=False),
                                        bottom=widgets.IntSlider(min=0,
                                                                 max=height - 1,
                                                                 value=bottom,
                                                                 continuous_update=False),
                                       )
            display(self.beam_fluctuation_roi)

        else:
            self.proj_norm_beam_fluctuation = self.proj_norm

    def export_normalization(self):
        working_dir = os.path.join(self.working_dir, "shared", "processed_data")
        if not os.path.exists(working_dir):
            working_dir = self.working_dir

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.export_normalized_data)
        list_folder_selected = o_file_browser.select_output_folder(instruction="Select output folder")

    def export_normalized_data(self, folder):
        print(f"New folder will be created in {folder} and called {self.input_folder_base_name}_YYYYMMDDHHMM")
        save_data(data=np.asarray(self.proj_norm),
                  outputbase=folder,
                  name=self.input_folder_base_name + "_normalized")

    def saving_beam_fluctuation_correction(self, background_region):
        self.background_region = background_region

    def beam_fluctuation_correction_embedded(self):
        if self.beam_fluctuation_ui.value:
            background_region = list(self.beam_fluctuation_roi.result)
            self._beam_fluctuation(background_region=background_region)

    def beam_fluctuation_correction(self):
        background_region = self.background_region
        self._beam_fluctuation(background_region=background_region)

    def _beam_fluctuation(self, background_region=None):

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
                                       figsize=(5, 10))
        # before beam fluctuation
        # proj_norm_min = np.min(proj_norm, axis=0)
        # fig0 = ax0.imshow(proj_norm_min)
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
        rot_angles = self.rot_angles

        # let's find where is index of the angle the closer to 180.0
        angles_minus_180 = rot_angles - 180.0
        abs_angles_minus_180 = np.abs(angles_minus_180)
        minimum_value = np.min(abs_angles_minus_180)

        index_0_degree = 0
        index_180_degree = np.where(minimum_value == abs_angles_minus_180)[0][0]

        rot_angles_sorted = rot_angles[:]
        rot_angles_sorted.sort()
        self.mean_delta_angle = np.mean([y - x for (x, y) in zip(rot_angles_sorted[:-1],
                                                            rot_angles_sorted[1:])])

        # list_180_deg_pairs_idx = tilt.find_180_deg_pairs_idx(angles=self.rot_angles,
        #                                                      atol=self.mean_delta_angle)
        #
        # index_0_degree = list_180_deg_pairs_idx[0][0]
        # index_180_degree = list_180_deg_pairs_idx[1][0]

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
                           self.tilt_checkbox4,
                           self.tilt_checkbox5]
        full_list_checkboxes = list_checkboxes[:]
        list_methods = [self.tilt_checkbox1_changed,
                        self.tilt_checkbox2_changed,
                        self.tilt_checkbox3_changed,
                        self.tilt_checkbox4_changed,
                        self.tilt_checkbox5_changed]

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
        """direct minimization"""
        self.tilt_checkbox_handler(checkbox_index=1)

    def tilt_checkbox2_changed(self, value):
        """phase correlation"""
        self.tilt_checkbox_handler(checkbox_index=2)

    def tilt_checkbox3_changed(self, value):
        """use center"""
        self.tilt_checkbox_handler(checkbox_index=3)

    def tilt_checkbox4_changed(self, value):
        """scipy minimizer"""
        self.tilt_checkbox_handler(checkbox_index=4)

    def tilt_checkbox5_changed(self, value):
        """user defined"""
        self.tilt_checkbox_handler(checkbox_index=5)

    def calculate_tilt(self):

        # find out index of 0 and 180 degrees images
        self.index_0_degree = self.left_select.index
        self.index_180_degree = self.right_select.index

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

        line2 = widgets.HBox([widgets.Checkbox(value=True,
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

        line3 = widgets.HBox([widgets.Checkbox(value=True,
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

        # scipy minimizer
        line4 = widgets.HBox([widgets.Checkbox(value=True,
                                               description="Scipy minimizer"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label("",
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.tilt_checkbox4 = line4.children[0]
        self.scipy_minimizer_value = line4.children[1]
        scipy_minimizer_status = line4.children[3]

        # user defined
        line5 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="User defined"),
                              widgets.FloatText(0,
                                                layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label("",
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.tilt_checkbox5 = line5.children[0]
        self.user_value = line5.children[1]

        # self.tilt_checkbox1.observe(self.tilt_checkbox1_changed, names="value")
        # self.tilt_checkbox2.observe(self.tilt_checkbox2_changed, names="value")
        # self.tilt_checkbox3.observe(self.tilt_checkbox3_changed, names="value")
        # self.tilt_checkbox4.observe(self.tilt_checkbox4_changed, names="value")
        # self.tilt_checkbox5.observe(self.tilt_checkbox5_changed, names="value")

        vertical_layout = widgets.VBox([line1, line2, line3, line4, line5])
        display(vertical_layout)

        # direct minimization
        o_direct = DirectMinimization(index_0_degree=self.index_0_degree,
                                      index_180_degree=self.index_180_degree,
                                      proj_mlog=self.proj_mlog)
        tilt_value1 = o_direct.compute()
        self.direct_minimization_value.value = f"{tilt_value1:.3f}"
        self.dict_tilt_values[TiltAlgorithms.direct_minimization] = tilt_value1
        direct_minimization_status.value = DONE
        phase_correlation_status.value = IN_PROGRESS

        # phase correlation
        o_phase = PhaseCorrelation(index_0_degree=self.index_0_degree,
                                   index_180_degree=self.index_180_degree,
                                   proj_mlog=self.proj_mlog)
        tilt_value2 = o_phase.compute()
        self.phase_correlation_value.value = f"{tilt_value2:.3f}"
        self.dict_tilt_values[TiltAlgorithms.phase_correlation] = tilt_value2
        phase_correlation_status.value = DONE
        use_center_status.value = IN_PROGRESS

        # use center
        o_center = UseCenter(index_0_degree=self.index_0_degree,
                             index_180_degree=self.index_180_degree,
                             proj_mlog=self.proj_mlog)
        tilt_value3 = o_center.compute()
        self.use_center_value.value = f"{tilt_value3:.3f}"
        self.dict_tilt_values[TiltAlgorithms.use_center] = tilt_value3
        use_center_status.value = DONE
        scipy_minimizer_status.value = IN_PROGRESS

        # scipy minimizer
        tilt_object = tilt.calculate_tilt(image0=self.proj_mlog[self.index_0_degree],
                                          image180=self.proj_mlog[self.index_180_degree])
        tilt_value4 = tilt_object.x
        self.scipy_minimizer_value.value = f"{tilt_value4:.3f}"
        self.dict_tilt_values[TiltAlgorithms.scipy_minimizer] = tilt_value4
        scipy_minimizer_status.value = DONE

    def get_tilt_value_selected(self):
        if self.tilt_checkbox1.value:
            return self.dict_tilt_values[TiltAlgorithms.direct_minimization]
        elif self.tilt_checkbox2.value:
            return self.dict_tilt_values[TiltAlgorithms.phase_correlation]
        elif self.tilt_checkbox3.value:
            return self.dict_tilt_values[TiltAlgorithms.use_center]
        elif self.tilt_checkbox4.value:
            return self.dict_tilt_values[TiltAlgorithms.scipy_minimizer]
        else:
            return self.user_value.value

    def apply_tilt_and_display(self):
        tilt_value = self.get_tilt_value_selected()
        print(f"Applying tilt correction using {tilt_value:.3f} ...")
        self.proj_tilt_corrected = tilt.apply_tilt_correction(arrays=self.proj_mlog,
                                                              tilt=tilt_value)

        fig, ax = plt.subplots(nrows=1, ncols=1, num="Tilt Correction", figsize=(10, 10))

        index_0_image = self.proj_tilt_corrected[self.index_0_degree]
        index_180_image_flipped = np.fliplr(self.proj_tilt_corrected[self.index_180_degree])
        self.overlap_image = np.add(index_0_image, index_180_image_flipped)/2.
        fig0 = ax.imshow(self.overlap_image)
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

    def filter_options(self):
        self.strikes_removal_option()
        self.remove_negative_values_option()

    def strikes_removal_option(self):
        self.strikes_removal_ui = widgets.Checkbox(value=False,
                                                   disabled= not enable_remove_ring_artifact,
                                                   description="Strikes removal")
        display(self.strikes_removal_ui)

    def remove_negative_values_option(self):
        self.remove_negative_ui = widgets.Checkbox(value=False,
                                                   description="Remove negative values")
        display(self.remove_negative_ui)

    def apply_filter_options(self):
        self.strikes_removal()
        self.remove_negative_values()

    def remove_negative_values(self):
        """remove all the intensity that are below 0"""
        if self.remove_negative_ui.value:
            self.proj_mlog[self.proj_mlog < 0] = 0
            print(" Removed negative values!")
        else:
            print(" Skipped remove negative values!")

    def strikes_removal(self):
        if self.strikes_removal_ui.value:
            t0 = timeit.default_timer()
            print("Running strikes removal ...")
            self.proj_strikes_removed = remove_ring_artifact(arrays=self.proj_tilt_corrected,
                                                             kernel_size=5,
                                                             max_workers=NCORE)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            self.proj_strikes_removed = self.proj_tilt_corrected
            print(" Skipped strikes removal!")

    def display_sinogram(self):
        self.sinogram_mlog = np.moveaxis(self.proj_strikes_removed, 1, 0)

        def plot_sinogram(index):
            fig, axis = plt.subplots(num="sinogram", figsize=(10, 10), nrows=1, ncols=1)
            axis.imshow(self.sinogram_mlog[index])
            axis.set_title(f"Sinogram at slice #{index}")

        plot_sinogram_ui = interactive(plot_sinogram,
                                       index=widgets.IntSlider(min=0,
                                                               max=len(self.sinogram_mlog),
                                                               value=0))
        display(plot_sinogram_ui)

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

    # testing the reconstruction on a few slices
    def define_slices_to_test_reconstruction(self):
        height, width = np.shape(self.overlap_image)
        nbr_slices = 4
        step = height / (nbr_slices + 1)
        slices = [k * step for k in np.arange(1, nbr_slices + 1)]

        display(
            HTML("<span style='color:blue'><b>Position of the slices you want to test the reconstruction with:</b>" +
                 "<br></span><b>To add a new slice</b>, enter value to the right of the last slice defined"))

        def display_image_and_slices(list_slices):
            fig, axs = plt.subplots(num='Select slices to reconstruct')
            fig.set_figwidth(15)
            axs.imshow(self.overlap_image)
            for _slice in list_slices:
                axs.axhline(_slice, color='red', linestyle='--')

            return list_slices

        self.display_slices = interactive(display_image_and_slices,
                                     list_slices=widgets.IntsInput(value=slices,
                                                                   min=0,
                                                                   max=height - 1))
        display(self.display_slices)

    def test_reconstruction(self):
        list_slices = self.display_slices.result
        rec_images = []
        for num, idx in enumerate(list_slices):
            rec_images.append(rec.gridrec_reconstruction(self.sinogram_mlog[idx],
                                                         self.rot_center[0],
                                                         angles=self.rot_angles,
                                                         apply_log=False,
                                                         ratio=1.0,
                                                         filter_name='shepp',
                                                         pad=100,
                                                         ncore=NCORE))

        # display slices reconstructed here
        def display_slices(slice_index):
            fig, axs = plt.subplots(num="testing reconstruction", ncols=2, nrows=1)
            fig.set_figwidth(15)

            axs[0].imshow(rec_images[slice_index])
            axs[0].set_title(f"Slice {list_slices[slice_index]}.")
            axs[1].imshow(self.overlap_image)
            axs[1].set_title(f"Integrated image and slice {list_slices[slice_index]} position.")

            axs[1].axhline(list_slices[slice_index], color='red', linestyle='--')

        display_test = interactive(display_slices,
                                   slice_index=widgets.IntSlider(min=0,
                                                                 max=len(list_slices) - 1,
                                                                 continuous_update=True))
        display(display_test)

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
            proj_mlog_bm3d = bm3d.extreme_streak_attenuation(self.proj_tilt_corrected)
            self.proj_ring_removal_1 = bm3d.multiscale_streak_removal(proj_mlog_bm3d)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using BM3D")
            self.proj_ring_removal_1 = self.proj_tilt_corrected

        # tomopy, Vo
        if self.ring_removal_ui.children[1].value:
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
        if self.ring_removal_ui.children[2].value:
            t0 = timeit.default_timer()
            print("Running strikes removal using Ketcham ...")
            self.proj_strikes_removed = remove_ring_artifact(arrays=self.proj_ring_removal_2,
                                                             kernel_size=5,
                                                             max_workers=NCORE)
            print(" strikes removal done!")
            t1 = timeit.default_timer()
            print(f"time= {t1 - t0:.2f}s")
        else:
            print("No strikes removal using Ketcham")
            self.proj_ring_removal_3 = self.proj_ring_removal_2

    def test_ring_removal(self):

        after_sinogram_mlog = self.proj_ring_removal_3.astype(np.float32)
        after_sinogram_mlog = np.moveaxis(after_sinogram_mlog, 1, 0)

        def plot_test_ring_removal(index):
            fig, axis = plt.subplots(num="sinogram", figsize=(15, 10), nrows=1, ncols=3)

            axis[0].imshow(self.sinogram_mlog[index])
            axis[0].set_title(f"Before ring removal")

            axis[1].imshow(after_sinogram_mlog[index])
            axis[1].set_title(f"After ring removal")

            axis[2].imshow(after_sinogram_mlog[index] - self.sinogram_mlog[index])
            axis[2].set_title(f"Difference")

        plot_test_ui = interactive(plot_test_ring_removal,
                                   index=widgets.IntSlider(min=0,
                                                           max=len(self.sinogram_mlog)))
        display(plot_test_ui)

    # Reconstruction
    def testing_reconstruction_algorithm(self):
        pass


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
        if not os.path.exists(working_dir):
            working_dir = self.working_dir

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.export_data)
        list_folder_selected = o_file_browser.select_output_folder(instruction="Select output folder")

    def export_data(self, folder):
        print(f"New folder will be created in {folder} and called {self.input_folder_base_name}_YYYYMMDDHHMM")
        save_data(data=np.asarray(self.reconstruction),
                  outputbase=folder,
                  name=self.input_folder_base_name)
