import os
import argparse
import json
import logging
import numpy as np
import copy
import gc

from imars3d.backend.dataio.data import load_data
from imars3d.backend.morph.crop import crop
from imars3d.backend.corrections.gamma_filter import gamma_filter
from imars3d.backend.preparation.normalization import normalization
from imars3d.backend.corrections.intensity_fluctuation_correction import normalize_roi
from imars3d.backend.diagnostics import tilt as diagnostics_tilt

from __code import DataType, BatchJsonKeys
from __code.utilities.system import retrieve_memory_usage


def load_json(json_file_name):
    """load the json and return the dict"""
    if not os.path.exists(json_file_name):
        raise FileNotFoundError(f"Config file {json_file_name} does not exist!")
    with open(json_file_name) as json_file:
        data = json.load(json_file)
    return data


def main(config_file_name):
    config_dict = load_json(config_file_name)

    # general
    number_of_cores = config_dict[BatchJsonKeys.number_of_cores]
    step_size = config_dict[BatchJsonKeys.step_size]

    # log file
    log_file_name = config_dict[BatchJsonKeys.log_file_name]
    logging.basicConfig(filename=log_file_name,
                        filemode='a',
                        format='[%(levelname)s] - %(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info("*** Starting the batch process ***")
    
    # load data
    list_raw_files = config_dict[BatchJsonKeys.list_raw_files]
    list_ob_files = config_dict[BatchJsonKeys.list_ob_files]
    list_dc_files = config_dict[BatchJsonKeys.list_dc_files]

    logging.info(f"Number of raw files: {len(list_raw_files)}")
    logging.info(f"Number of ob: {len(list_ob_files)}")
    logging.info(f"Number of dc: {len(list_dc_files)}")
    logging.info(f"Memory usage: {retrieve_memory_usage()}")
    
    logging.info(f"Loading the files ...")
    proj, ob, dc, rot_angles = load_data(ct_files=list_raw_files,
                                         ob_files=list_ob_files,
                                         dc_files=list_dc_files,
                                         max_workers=number_of_cores)
    if config_dict[BatchJsonKeys.select_dc_flag]:
        dc = np.array([np.zeros_like(proj[0])])
    logging.info(f"Loading done!")
    logging.info(f"Memory usage: {retrieve_memory_usage()}")

    # copy of raw data
    copy_proj = copy.deepcopy(proj)

    # crop region
    [left, right, top, bottom] = config_dict[BatchJsonKeys.crop_region]
    logging.info(f"Cropping using {left =}, {right =}, {top =}, {bottom =} ...")
    proj = crop(arrays=proj,
                crop_limit=crop_region)
    ob = crop(arrays=ob,
              crop_limit=crop_region)
    dc = crop(arrays=dc,
              crop_limit=crop_region)
    logging.info(f"Cropping done!")
    logging.info(f"Memory usage: {retrieve_memory_usage()}")

    # gamma filtering
    if config_dict[BatchJsonKeys.gamma_filtering_flag]:
        logging.info(f"Gamma filtering ...")
        number_of_projections = proj.shape[0]
        cores_times_step = number_of_cores * step_size
        for i in np.arange(0, number_of_projections, cores_times_step):
            end_idx = min(i + cores_times_step, number_of_projections)
            proj[i:end_idx] = gamma_filter(
            arrays=proj[i:end_idx],
            selective_median_filter=False,
            diff_tomopy=20,
            max_workers=number_of_cores,
            median_kernel=3,
            )
        logging.info(f"Gamma filtering Done!")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f"Gamma filtering skipped!")

    # normalization
    logging.info(f"Normalization ...")
    for i in np.arange(0, number_of_projections, step_size):
        end_idx = min(i + step_size, number_of_projections)
        proj[i:end_idx] = normalization(
            arrays=proj[i:end_idx],
            flats=ob,
            darks=dc,
            max_workers=number_of_cores
            )
        ob = None
        dc = None
        gc.collect()
    logging.info(f"Normalization done!")
    logging.info(f"Memory usage: {retrieve_memory_usage()}")

    # beam flucuation
    if config_dict[BatchJsonKeys.beam_fluctuation_flag]:
        logging.info(f"Beam fluctuation correction ...")
        [left, right, top, bottom] = config_dict[BatchJsonKeys.beam_fluctuation_region]
        logging.info(f" region selected: {left =}, {right =}, {top =}, {bottom =}")
        roi = [top, left, bottom, right]
        for i in np.arange(0, number_of_projections, step_size):
            end_idx = min(i + step_size, number_of_projections)
            proj[i: end_idx] = normalize_roi(
                                            ct=proj[i: end_idx],
                                            roi=roi,
                                            max_workers=number_of_cores,
                                        )
        logging.info(f"Beam fluctuation correction done!")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f"Beam fluctuation correction skipped!")

    # tilt_calculation
    tilt_value = config_dict[BatchJsonKeys.tilt_value]
    if tilt_value != 0.0:
        logging.info(f"Applying tilt {tilt_value} degrees ...")
        proj = diagnostics_tilt.apply_tilt_correction(arrays=proj, tilt=tilt_value)
        logging.info(f"Applying tilt done!")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f"Tilt correction skipped (tilt value is {tilt_value})!")



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
