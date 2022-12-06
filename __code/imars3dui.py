import os
import glob

from imars3d.backend.dataio.data import load_data, _get_filelist_by_dir

from __code.file_folder_browser import FileFolderBrowser


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

    def load_data(self):
        self.proj_raw, self.ob_raw, self.dc_raw, self.rot_angles = load_data(ct_files=self.input_files[DataType.raw],
                                                                             ob_files=self.input_files[DataType.ob],
                                                                             dc_files=self.input_files[DataType.dc])