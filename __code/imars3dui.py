import os
import glob
import matplotlib.pyplot as plt
import numpy as np

from imars3d.backend.dataio.data import load_data, _get_filelist_by_dir
from imars3d.backend.morph.crop import crop
from imars3d.backend.corrections.gamma_filter import gamma_filter
from imars3d.backend.preparation.normalization import normalization
import tomopy

from __code.file_folder_browser import FileFolderBrowser
from __code import DEFAULT_CROP_ROI, NCORE, DEFAULT_BACKROUND_ROI


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
        working_dir = os.path.join(self.working_dir, default_input_folder[data_type])

        o_file_browser = FileFolderBrowser(working_dir=working_dir,
                                           next_function=self.data_selected)
        list_folder_selected = o_file_browser.select_input_folder(instruction=
                                                                  f"Select Folder of {data_type}",
                                                                  multiple_flag=multiple_flag)

    def data_selected(self, list_folders):
        self.input_data_folders[self.current_data_type] = list_folders

        if self.current_data_type == DataType.raw:
            list_folders = [list_folders]

        list_files = self.retrieve_list_of_files(list_folders)
        self.input_files[self.current_data_type] = list_files

        print(f"{self.current_data_type} folder selected: {list_folders} with {len(list_files)} files)")

    def retrieve_list_of_files(self, list_folders):
        list_files = []
        for _folder in list_folders:
            _tiff_files = glob.glob(os.path.join(_folder, "*.tif*"))
            list_files = [*list_files, *_tiff_files]

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
        ob_min = np.min(self.ob_raw, axis=0)
        dc_max = np.max(self.dc_raw, axis=0)

        plt0 = ax0.imshow(proj_min)
        fig.colorbar(plt0, ax=ax0)
        ax0.set_title("np.min(proj_raw)")

        plt1 = ax1.imshow(ob_min)
        fig.colorbar(plt1, ax=ax1)
        ax1.set_title("np.min(ob_raw)")

        plt2 = ax2.imshow(dc_max)
        fig.colorbar(plt2, ax=ax2)
        ax2.set_title("np.min(dc_raw)")

        fig.tight_layout()

    def crop_and_display_data(self, crop_region):

        self.proj_crop = crop(arrays=self.proj_raw,
                              crop_limit=crop_region)
        self.ob_crop = crop(arrays=self.ob_raw,
                            crop_limit=crop_region)
        self.dc_crop = crop(arrays=self.dc_raw,
                            crop_limit=crop_region)

        self.proj_crop_min = crop(arrays=self.proj_min,
                                  crop_limit=crop_region)

        # fig, ax0 = plt.subplots(nrows=1, ncols=1,
        #                         figsize=(7, 5),
        #                         num="Cropped")
        #
        # fig1 = ax0.imshow(self.proj_crop_min)
        # plt.colorbar(fig1, ax=ax0)
        # ax0.set_title("min of proj")

    def gamma_filtering(self):
        self.proj_gamma = gamma_filter(arrays=self.proj_crop.astype(np.uint16),
                                       selective_median_filter=False,
                                       diff_tomopy=20,
                                       max_workers=NCORE,
                                       median_kernel=3)

    def normalization_and_display(self):
        self.proj_norm = normalization(arrays=self.proj_gamma,
                                       flats=self.ob_crop,
                                       darks=self.dc_crop)

        plt.figure()
        proj_norm_min = np.min(self.proj_norm, axis=0)
        plt.imshow(proj_norm_min)
        plt.colorbar()

    def beam_fluctuation_correction(self, background_region):
        roi = [background_region[2], background_region[0],
               background_region[3], background_region[1]]

        self.proj_norm_beam_fluctuation = tomopy.prep.normalize.normalize_roi(
            self.proj_norm,
            roi=roi,
            ncore=NCORE)

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
