import glob
import os
import json


@staticmethod
def retrieve_list_of_files(list_folders):
    list_files = []
    for _folder in list_folders:
        _tiff_files = glob.glob(os.path.join(_folder, "*.tif*"))
        list_files = [*list_files, *_tiff_files]

    list_files.sort()
    return list_files


@staticmethod
def load_json(json_file_name):
    if not os.path.exists(json_file_name):
        return None

    with open(json_file_name) as json_file:
        data = json.load(json_file)

    return data


@staticmethod
def save_json(json_file_name, json_dictionary=None):
    with open(json_file_name, 'w') as outfile:
        json.dump(json_dictionary, outfile)
