import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import timeit
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML

import warnings
warnings.filterwarnings('ignore')

from imars3d.backend.dataio.data import load_data
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
    crop_roi = DEFAULT_CROP_ROI
    background_roi = DEFAULT_BACKROUND_ROI

    dict_tilt_values = {}

    # name of raw folder (used to customize output file name)
    input_folder_base_name = None

    # data arrays
    proj_raw = None
    ob_raw = None
    dc_raw = None

    mean_delta_angle = None

    def __init__(self, working_dir="./"):
        self.working_dir = os.path.join(working_dir, 'raw', default_input_folder[DataType.raw])

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

        if not os.path.exists(self.working_dir):
            self.working_dir = os.path.abspath(os.path.expanduser("~"))

        o_file_browser = FileFolderBrowser(working_dir=self.working_dir,
                                           next_function=self.data_selected)
        o_file_browser.select_input_folder(instruction=f"Select Folder of {data_type}",
                                           multiple_flag=multiple_flag)

    def data_selected(self, list_folders):
        self.input_data_folders[self.current_data_type] = list_folders

        if self.current_data_type == DataType.raw:
            list_folders = [os.path.abspath(list_folders)]
            self.working_dir = os.path.dirname(os.path.dirname(list_folders[0]))  # default folder is the parent folder of sample
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
        self.proj, self.ob, self.dc, self.rot_angles = load_data(ct_files=self.input_files[DataType.raw],
                                                                 ob_files=self.input_files[DataType.ob],
                                                                 dc_files=self.input_files[DataType.dc])

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(5, 9))
        proj_min = np.min(self.proj, axis=0)
        self.proj_min = proj_min
        ob_max = np.max(self.ob, axis=0)
        dc_max = np.max(self.dc, axis=0)

        plt0 = ax0.imshow(proj_min)
        fig.colorbar(plt0, ax=ax0)
        ax0.set_title("np.min(proj)")

        plt1 = ax1.imshow(ob_max)
        fig.colorbar(plt1, ax=ax1)
        ax1.set_title("np.max(ob)")

        plt2 = ax2.imshow(dc_max)
        fig.colorbar(plt2, ax=ax2)
        ax2.set_title("np.max(dc)")

        fig.tight_layout()

    def crop_embedded(self):
        list_images = self.proj
        integrated_image = np.mean(list_images, axis=0)
        height, width = np.shape(integrated_image)

        def plot_crop(left, right, top, bottom):

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            ax.imshow(integrated_image)

            ax.axvline(left, color='blue', linestyle='--')
            ax.axvline(right, color='red', linestyle='--')

            ax.axhline(top, color='blue', linestyle='--')
            ax.axhline(bottom, color='red', linestyle='--')

            return left, right, top, bottom

        self.cropping = interactive(plot_crop,
                                    left=widgets.IntSlider(min=0,
                                                           max=width - 1,
                                                           value=0,
                                                           continuous_update=True),
                                    right=widgets.IntSlider(min=0,
                                                            max=width - 1,
                                                            value=width - 1,
                                                            continuous_update=False),
                                    top=widgets.IntSlider(min=0,
                                                          max=height - 1,
                                                          value=0,
                                                          continuous_update=False),
                                    bottom=widgets.IntSlider(min=0,
                                                             max=height - 1,
                                                             value=height - 1,
                                                             continuous_update=False),
                                    )
        display(self.cropping)

    def saving_crop_region(self, crop_region):
        self.crop_region = crop_region

    def perform_embedded_cropping(self):
        crop_region = list(self.cropping.result)
        self._crop_region(crop_region=crop_region)

    def _crop_region(self, crop_region):
        print(f"Running crop ...")
        self.proj = crop(arrays=self.proj,
                         crop_limit=crop_region)
        self.ob = crop(arrays=self.ob,
                       crop_limit=crop_region)
        self.dc = crop(arrays=self.dc,
                       crop_limit=crop_region)

        self.proj_min = crop(arrays=self.proj_min,
                             crop_limit=crop_region)
        print(f"cropping done!")

    def perform_cropping(self):
        crop_region = self.crop_region
        self._crop_region(crop_region=crop_region)

    def gamma_filtering(self):
        print(f"Running gamma filtering ...")
        t0 = timeit.default_timer()
        self.proj = gamma_filter(arrays=self.proj.astype(np.uint16),
                                 selective_median_filter=False,
                                 diff_tomopy=20,
                                 max_workers=NCORE,
                                 median_kernel=3)
        t1 = timeit.default_timer()
        print(f"Gamma filtering done in {t1-t0:.2f}s")

    def normalization_and_display(self):
        print(f"Running normalization ...")
        t0 = timeit.default_timer()
        self.proj_norm = normalization(arrays=self.proj,
                                       flats=self.ob,
                                       darks=self.dc)
        t1 = timeit.default_timer()
        print(f"normalization done in {t1-t0:.2f}s")

        plt.figure()
        proj_norm_min = np.min(self.proj_norm, axis=0)
        plt.imshow(proj_norm_min)
        plt.colorbar()

    def select_beam_fluctuation_roi_embedded(self):
        integrated_image = np.mean(self.proj_norm, axis=0)
        height, width = np.shape(integrated_image)

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
                                                           value=0,
                                                           continuous_update=True),
                                    right=widgets.IntSlider(min=0,
                                                            max=width - 1,
                                                            value=width - 1,
                                                            continuous_update=False),
                                    top=widgets.IntSlider(min=0,
                                                          max=height - 1,
                                                          value=0,
                                                          continuous_update=False),
                                    bottom=widgets.IntSlider(min=0,
                                                             max=height - 1,
                                                             value=height - 1,
                                                             continuous_update=False),
                                   )
        display(self.beam_fluctuation_roi)

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

        import copy
        self.proj_norm_before = copy.deepcopy(self.proj_norm)
        self.proj_norm = normalize_roi(
                ct=self.proj_norm,
                roi=roi,
                max_workers=NCORE)

        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,
                                       num="Beam fluctuation",
                                       figsize=(5, 10))
        # before beam fluctuation
        # proj_norm_min = np.min(proj_norm, axis=0)
        # fig0 = ax0.imshow(proj_norm_min)
        fig0 = ax0.imshow(self.proj_norm_before[0])
        ax0.set_title("before")
        plt.colorbar(fig0, ax=ax0)

        # after beam fluctuation
        # proj_norm_beam_fluctuation_min = np.min(proj_norm_beam_fluctuation, axis=0)
        # fig1 = ax1.imshow(proj_norm_beam_fluctuation_min)
        fig1 = ax1.imshow(self.proj_norm[0])
        ax1.set_title("after")
        plt.colorbar(fig1, ax=ax1)

    def minus_log_and_display(self):
        self.proj_mlog = tomopy.minus_log(self.proj_norm)
        try:
            del self.proj_norm_before
        except AttributeError:
            pass

        del self.proj_norm
        plt.figure()
        plt.imshow(self.proj_mlog[0])
        plt.colorbar()

    def calculate_mean_delta_angle(self):
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

    def find_0_180_degrees_files(self):
        self.calculate_mean_delta_angle()

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

        # scipy minimizer
        line4 = widgets.HBox([widgets.Checkbox(value=False,
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

        self.tilt_checkbox1.observe(self.tilt_checkbox1_changed, names="value")
        self.tilt_checkbox2.observe(self.tilt_checkbox2_changed, names="value")
        self.tilt_checkbox3.observe(self.tilt_checkbox3_changed, names="value")
        self.tilt_checkbox4.observe(self.tilt_checkbox4_changed, names="value")
        self.tilt_checkbox5.observe(self.tilt_checkbox5_changed, names="value")

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
        self.proj_mlog = tilt.apply_tilt_correction(arrays=self.proj_mlog,
                                                    tilt=tilt_value)

        fig, ax = plt.subplots(nrows=1, ncols=1, num="Tilt Correction", figsize=(10, 10))

        index_0_image = self.proj_mlog[self.index_0_degree]
        index_180_image_flipped = np.fliplr(self.proj_mlog[self.index_180_degree])
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
        self.proj_mlog = remove_ring_artifact(arrays=self.proj_mlog,
                                                         kernel_size=5,
                                                         max_workers=NCORE)
        print(" strikes removal done!")
        t1 = timeit.default_timer()
        print(f"time= {t1 - t0:.2f}s")

    def display_sinogram(self):

        fig, axis = plt.subplots(num="sinogram", figsize=(5, 5), nrows=1, ncols=1)
        sinogram_mlog = np.moveaxis(self.proj_mlog, 1, 0)

        def plot_sinogram(index):

            axis.imshow(sinogram_mlog[index])
            axis.set_title(f"Sinogram at slice #{index}")

        plot_sinogram_ui = interactive(plot_sinogram,
                                       index=widgets.IntSlider(min=0,
                                                               max=len(sinogram_mlog),
                                                               value=0))
        display(plot_sinogram_ui)


    def rotation_center(self):

        if not self.mean_delta_angle:
            self.calculate_mean_delta_angle()

        print(f"Running rotation center ...")
        t0 = timeit.default_timer()
        self.rot_center = find_rotation_center(arrays=self.proj_mlog,
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

        self.reconstruction = recon(arrays=self.proj_mlog,
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
