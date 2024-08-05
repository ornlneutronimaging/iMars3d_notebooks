import os
import argparse
import json

from __code import DataType, BatchJsonKeys


def load_json(json_file_name):
    """load the json and return the dict"""
    if not os.path.exists(json_file_name):
        raise FileNotFoundError(f"Config file {json_file_name} does not exist!")
    with open(json_file_name) as json_file:
        data = json.load(json_file)
    return data


def main(config_file_name):
    config_dict = load_json(config_file_name)

    # load data
    list_raw_files = config_dict[BatchJsonKeys.list_raw_files]
    list_ob_files = config_dict[BatchJsonKeys.list_ob_files]
    list_dc_files = config_dict[BatchJsonKeys.list_dc_files]

    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''Run laminography
                                     
                                     example:
                                          laminography_cli config_file.json
                                     ''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('config_file',
                      help='Full path to config file name')
    args = parser.parse_args()

    config_file_name = args.config_file

    main(config_file_name)
