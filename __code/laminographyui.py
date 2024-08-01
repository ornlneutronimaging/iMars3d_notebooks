import os
import psutil
import numpy as np
import time
import ipywidgets as widgets
from IPython.display import display

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

from __code import DataType, TiltAlgorithms, TiltTestKeys, config

from __code.workflow.load import Load
from __code.workflow.crop import Crop
from __code.workflow.gamma_filtering import GammaFiltering
from __code.workflow.normalization import Normalization
from __code.workflow.beam_fluctuation_correction import BeamFluctuationCorrection
from __code.workflow.transmission_to_attenuation import TransmissionToAttenuation
from __code.workflow.tilt import Tilt
from __code.workflow.reconstruction import TestReconstruction
from __code.workflow.ring_removal import RingRemoval
from __code.workflow.filters import Filters
from __code.workflow.sinogram import Sinogram
from __code.workflow.select_z_range import SelectZRange
from __code.laminography_event_handler import LaminographyEventHandler

from __code.file_folder_browser import FileFolderBrowser
from __code.display import Display

default_input_folder = {DataType.raw: 'ct_scans',
                        DataType.ob: 'ob',
                        DataType.dc: 'dc'}


class LaminographyUi:

    working_dir = {DataType.raw: "",
                   DataType.ob: "",
                   DataType.dc: "",
                   }

    input_data_folders = {}
    input_files = {}

    untouched_sample_data = None

    rot_angles_rad = None
    rot_angles_deg = None

    test_tilt_reconstruction = {TiltAlgorithms.phase_correlation: None,
                                TiltAlgorithms.direct_minimization: None,
                                TiltAlgorithms.use_center: None,
                                TiltAlgorithms.scipy_minimizer: None,
                                TiltAlgorithms.user: None,
                                }

    if config.debugging:
        crop_roi = config.DEFAULT_CROP_ROI
        background_roi = config.DEFAULT_BACKROUND_ROI
        test_tilt_slices = config.DEFAULT_TILT_SLICES_SELECTION
    else:
        crop_roi = [None, None, None, None]
        background_roi = [None, None, None, None]
        test_tilt_slices = [None, None]

    dict_tilt_values = {}

    # name of raw folder (used to customize output file name)
    input_folder_base_name = None

    select_dc_flag = None  # use or not dc files

    # data arrays
    proj_raw = None
    ob_raw = None
    dc_raw = None

    # name of the raw folder
    raw_folder_name = None

    investigate_data_flag = False

    o_tilt = None
    o_test_reconstruction = None
    o_ring_removal = None

    # Ring removal
    sinogram_before_ring_removal = None
    sinogram_after_ring_removal = None

    # final reconstruction volume
    recon_mbir = None

    def __init__(self, working_dir="./"):
        init_path_to_raw = os.path.join(working_dir, 'raw')
        self.working_dir[DataType.ipts] = working_dir
        self.working_dir[DataType.raw] = os.path.join(init_path_to_raw, default_input_folder[DataType.raw])
        self.working_dir[DataType.ob] = os.path.join(init_path_to_raw, default_input_folder[DataType.ob])
        self.working_dir[DataType.dc] = os.path.join(init_path_to_raw, default_input_folder[DataType.dc])
        print("version 07-30-2024")

    # SELECT INPUT DATA ===============================================================================================
    def select_raw(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.raw)

    def select_ob(self):
        o_load = Load(parent=self)
        o_load.select_folder(data_type=DataType.ob,
                             multiple_flag=True)

    def select_dc_options(self):
        o_load = Load(parent=self)
        o_load.select_dc_options()

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

    # INVESTIGATE DATA ====================================================================================
    
    def investigate_loaded_data_flag(self):
        o_event = Load(parent=self)
        o_event.investigate_loaded_data_flag()

    def investigate_loaded_data(self):
        o_event = Load(parent=self)
        o_event.investigate_loaded_data()

    # CROP ===============================================================================================

    def crop_embedded(self):
        o_crop = Crop(parent=self)
        o_crop.crop_embedded()

    def saving_crop_region(self, crop_region):
        self.crop_region = crop_region

    def perform_embedded_cropping(self):
        crop_region = list(self.cropping.result)
        self.crop_region = crop_region
        o_crop = Crop(parent=self)
        o_crop.crop_region(crop_region)

    # GAMMA_FILTERING =====================================================================================

    def gamma_filtering_options(self):
        o_gamma = GammaFiltering(parent=self)
        o_gamma.gamma_filtering_options()

    def gamma_filtering(self):
        o_gamma = GammaFiltering(parent=self)
        o_gamma.gamma_filtering()

    # NORMALIZATION =====================================================================================

    def normalization_and_display(self):
        o_norm = Normalization(parent=self)
        o_norm.normalization_and_display()

    def export_normalization(self):
        o_norm = Normalization(parent=self)
        o_norm.export_normalization()

    # BEAM FLUCTUATION =====================================================================================

    def beam_fluctuation_correction_option(self):
        o_beam = BeamFluctuationCorrection(parent=self)
        o_beam.beam_fluctuation_correction_option()

    def apply_select_beam_fluctuation(self):
        o_beam = BeamFluctuationCorrection(parent=self)
        o_beam.apply_select_beam_fluctuation()

    def define_beam_fluctuation_settings(self):
        o_beam = BeamFluctuationCorrection(parent=self)
        o_beam.apply_select_beam_fluctuation()

    def beam_fluctuation_correction_embedded(self):
        o_beam = BeamFluctuationCorrection(parent=self)
        o_beam.beam_fluctuation_correction_embedded()

    def run_beam_fluctuation_correction(self):
        o_beam = BeamFluctuationCorrection(parent=self)
        o_beam.beam_fluctuation_correction_embedded()

    def saving_beam_fluctuation_correction(self, background_region):
        self.background_region = background_region

    def beam_fluctuation_correction(self):
        background_region = self.background_region
        self._beam_fluctuation(background_region=background_region)

    # TRANSMISSION TO ATTENUATION ===========================================================================

    def minus_log_and_display(self):
        o_trans = TransmissionToAttenuation(parent=self)
        o_trans.minus_log_and_display()

    # TILT CORRECTION =======================================================================================

    def find_0_180_degrees_files(self):
        o_tilt = Tilt(parent=self)
        o_tilt.find_0_180_degrees_files()

    def calculate_tilt(self):
        self.o_tilt = Tilt(parent=self)
        self.o_tilt.calculate_tilt()

    # def apply_tilt_and_display(self):
    #     o_tilt = Tilt(parent=self)
    #     o_tilt.apply_tilt()
    #     o_tilt.display_tilt()

    def test_tilt_slices_selection(self):
        self.o_tilt.test_slices_selection()

    def testing_tilt_on_selected_algorithms(self):
        self.o_tilt.apply_tilt_using_selected_algorithms()
        self.o_tilt.perform_reconstruction_on_selected_data_sets()
        self.o_tilt.display_test_results()

    def display_with_tilt(self):
        algo_selected = self.o_tilt.test_tilt.result
        self.tilt_algo_selected_finally = algo_selected
        self.proj_tilt_corrected = self.o_tilt.test_tilt_reconstruction[algo_selected][TiltTestKeys.raw_3d]
        self.rot_center = self.o_tilt.test_tilt_reconstruction[algo_selected][TiltTestKeys.center_of_rotation]
        self.o_tilt.display_tilt()

    # FILTERING ==========================================================================================

    def filter_options(self):
        # self.strikes_removal_option()
        self.remove_negative_values_option()

    def strikes_removal_option(self):
        self.strikes_removal_ui = widgets.Checkbox(value=False,
                                                   disabled=not enable_remove_ring_artifact,
                                                   description="Strikes removal")
        display(self.strikes_removal_ui)

    def remove_negative_values_option(self):
        self.remove_negative_ui = widgets.Checkbox(value=False,
                                                   description="Remove negative values")
        display(self.remove_negative_ui)

    def apply_filter_options(self):
        # self.strikes_removal()
        self.remove_negative_values()

    def remove_negative_values(self):
        o_filter = Filters(parent=self)
        o_filter.remove_negative_values()

    def strikes_removal(self):
        o_filter = Filters(parent=self)
        o_filter.strikes_removal()

    # SINOGRAM ==============================================================================================

    def create_and_display_sinogram(self):
        sinogram_data = Sinogram.create_sinogram(data_3d=self.proj_tilt_corrected)
        o_display = Display(parent=self)
        o_display.sinogram(sinogram_data=sinogram_data)
        self.sinogram_before_ring_removal = sinogram_data

    # ROTATION CENTER =======================================================================================

    def rotation_center(self):
        print(f"Running rotation center ...")
        # t0 = timeit.default_timer()
        t0 = time.time()
        self.rot_center = find_rotation_center(arrays=self.proj_tilt_corrected,
                                               angles=self.rot_angles,
                                               num_pairs=-1,
                                               in_degrees=True,
                                               atol_deg=self.mean_delta_angle,
                                               )
        # t1 = timeit.default_timer()
        t1 = time.time()
        print(f"rotation center found in {t1-t0:.2f}s")
        print(f" - value: {self.rot_center}")

    # RING REMOVAL ==========================================================================================

    # will crete the proj_ring_removed data set

    def ring_removal_options(self):
        self.o_ring_removal = RingRemoval(parent=self)
        self.o_ring_removal.ring_removal_options()

    def apply_ring_removal_options(self):
        self.o_ring_removal.apply_ring_removal_options()

    def test_ring_removal(self):
        self.o_ring_removal.test_ring_removal()

    # TEST RECONSTRUCTION ====================================================================================

    # testing the reconstruction on a few slices
    def define_slices_to_test_reconstruction(self):
        self.o_test_reco = TestReconstruction(parent=self)
        self.o_test_reco.define_slices_to_test_reconstruction()

    def test_reconstruction(self):
        self.sinogram_mlog = Sinogram.create_sinogram(data_3d=self.proj_ring_removed)
        self.o_test_reco.test_reconstruction()

    # RECONSTRUCTION  ==========================================================================================

    def testing_reconstruction_algorithm(self):
        self.o_test_reco.testing_reconstruction_algorithms()

    def running_reconstruction_test(self):
        self.o_test_reco.retrieving_parameters()
        self.o_test_reco.running_reconstruction_test()

    def select_range_of_slices(self):
        o_select = SelectZRange(parent=self)
        o_select.select_range_of_slices()

    def display_reconstruction_test(self):
        pass

    # Laminography
    def laminography_settings(self):
        self.o_event_laminography_settings = LaminographyEventHandler(parent=self)
        self.o_event_laminography_settings.set_settings()

    def run_laminography(self):
        self.o_event_laminography_settings.run()

    def visualize_reconstruction(self):
        self.o_event_laminography_settings.visualize()

    # def reconstruction_and_display(self):
    #     t0 = timeit.default_timer()
    #     print("Running reconstruction ...")

    #     # converting angles from deg to radians
    #     rot_ang_rad = np.radians(self.rot_angles)

    #     self.reconstruction = recon(arrays=self.proj_strikes_removed,
    #                                 center=self.rot_center[0],
    #                                 theta=rot_ang_rad,
    #                                 )

    #     print(" reconstruction done!")
    #     t1 = timeit.default_timer()
    #     print(f"time= {t1 - t0:.2f}s")

    #     plt.figure()
    #     plt.imshow(self.reconstruction[0])
    #     plt.colorbar()
    #     plt.show()

    #     def plot_reconstruction(index):
    #         plt.title(f"Reconstructed slice #{index}")
    #         plt.imshow(self.reconstruction[index])
    #         plt.show()

    #     plot_reconstruction_ui = interactive(plot_reconstruction,
    #                                          index=widgets.IntSlider(min=0,
    #                                                                  max=len(self.reconstruction),
    #                                                                  value=0))
    #     display(plot_reconstruction_ui)

    def export(self):
        working_dir = os.path.join(self.working_dir[DataType.ipts], "shared", "processed_data")
        if not os.path.exists(working_dir):
            working_dir = self.working_dir

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.export_data)
        list_folder_selected = o_file_browser.select_output_folder(instruction="Select output folder")

    def export_data(self, folder):
        print(f"New folder will be created in {folder} and called {self.input_folder_base_name}_YYYYMMDDHHMM")
        save_data(data=np.asarray(self.recon_mbir),
                  outputbase=folder,
                  name=self.input_folder_base_name)
        print(f"Done!")

    def get_memory_usage(self):
        """
        Calculate the total memory usage of the current process and its children.

        Returns
        -------
        float
            Total memory usage in MB.
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_usage_mb = mem_info.rss / (1024 ** 2)  # Convert bytes to MB

        # Include memory usage of child processes
        for child in process.children(recursive=True):
            try:
                mem_info = child.memory_info()
                mem_usage_mb += mem_info.rss / (1024 ** 2)
            except psutil.NoSuchProcess:
                continue

        return mem_usage_mb

    def print_memory_usage(self):
        """
        Print the total memory usage.
        """
        mem_usage = self.get_memory_usage()
        print(f"Total memory usage: {mem_usage:.2f} MB")
