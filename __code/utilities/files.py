import glob
import os


@staticmethod
def retrieve_list_of_files(list_folders):
    list_files = []
    for _folder in list_folders:
        _tiff_files = glob.glob(os.path.join(_folder, "*.tif*"))
        list_files = [*list_files, *_tiff_files]

    list_files.sort()
    return list_files
