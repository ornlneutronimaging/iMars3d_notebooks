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

from __code import DataType, TiltAlgorithms
from __code.workflow.load import Load
from __code.workflow.crop import Crop
from __code.workflow.gamma_filtering import GammaFiltering
from __code.workflow.normalization import Normalization
from __code.workflow.beam_fluctuation_correction import BeamFluctuationCorrection
from __code.workflow.transmission_to_attenuation import TransmissionToAttenuation
from __code.workflow.tilt import Tilt

from __code.tilt.direct_minimization import DirectMinimization
from __code.tilt.phase_correlation import PhaseCorrelation
from __code.tilt.use_center import UseCenter
from __code import config

import tomopy

from __code.file_folder_browser import FileFolderBrowser
from __code import NCORE

default_input_folder = {DataType.raw: 'ct_scans',
                        DataType.ob: 'ob',
                        DataType.dc: 'dc'}


class Imars3dui:

    input_data_folders = {}
    input_files = {}

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

    # data arrays
    proj_raw = None
    ob_raw = None
    dc_raw = None

    o_tilt = None

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

    def beam_fluctuation_correction_embedded(self):
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

    def apply_tilt_and_display(self):
        o_tilt = Tilt(parent=self)
        o_tilt.apply_tilt()
        o_tilt.display_tilt()

    def test_tilt_slices_selection(self):
        self.o_tilt.test_slices_selection()

    def testing_tilt_on_selected_algorithms(self):
        self.o_tilt.apply_tilt_using_selected_algorithms()
        self.o_tilt.perform_reconstruction_on_selected_data_sets()
        self.o_tilt.display_test_results()











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
