import os
import argparse
import json
import logging
import numpy as np
import copy
import gc
import tomopy

from imars3d.backend.dataio.data import load_data
from imars3d.backend.morph.crop import crop
from imars3d.backend.corrections.gamma_filter import gamma_filter
from imars3d.backend.preparation.normalization import normalization
from imars3d.backend.corrections.intensity_fluctuation_correction import normalize_roi
from imars3d.backend.diagnostics import tilt as diagnostics_tilt
from imars3d.backend.corrections.ring_removal import remove_ring_artifact
from imars3d.backend.diagnostics.rotation import find_rotation_center
from tomoORNL.reconEngine import MBIR

from __code import DataType, BatchJsonKeys
from __code.utilities.system import retrieve_memory_usage


def load_json(json_file_name):
    """load the json and return the dict"""
    if not os.path.exists(json_file_name):
        raise FileNotFoundError(f"Config file {json_file_name} does not exist!")
    with open(json_file_name) as json_file:
        data = json.load(json_file)
    return data

def get_gpu_index(children_gpus_ui):
    gpu_index = []
    for _index, _child in enumerate(children_gpus_ui):
        if _child.value:
            gpu_index.append(_index)
    return gpu_index
    
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
    list_raw_files = config_dict[BatchJsonKeys.list_raw_files]   #FOR DEBUGGING ONLY, I only load the first 20
    list_ob_files = config_dict[BatchJsonKeys.list_ob_files]
    list_dc_files = config_dict[BatchJsonKeys.list_dc_files]

    logging.info(f"Number of raw files: {len(list_raw_files)}")
    logging.info(f"Number of ob: {len(list_ob_files)}")
    logging.info(f"Number of dc: {len(list_dc_files)}")
    logging.info(f"Memory usage: {retrieve_memory_usage()}")
    
    # checking if files are on the fastdata folder
    if os.path.exists(f"/fastdata/{list_raw_files[0]}"):
        print("yes!!!!!, it's in /fastdata")
        list_raw_files = [f"/fastdata/{_file}" for _file in list_raw_files]
    if os.path.exists(f"/fastdata/{list_ob_files[0]}"):
        list_ob_files = [f"/fastdata/{_file}" for _file in list_ob_files]
    if os.path.exists(f"/fastdata/{list_dc_files[0]}"):
        list_dc_files = [f"/fastdata/{_file}" for _file in list_dc_files]

    logging.info(f"Loading the files ...")
    proj, ob, dc, rot_angles = load_data(ct_files=list_raw_files,
                                         ob_files=list_ob_files,
                                         dc_files=list_dc_files,
                                         max_workers=number_of_cores)
    
    rot_angles_sorted = rot_angles[:]
    rot_angles_sorted.sort()
    mean_delta_angle = np.mean([y - x for (x, y) in zip(rot_angles_sorted[:-1],
                                                        rot_angles_sorted[1:])])

    if config_dict[BatchJsonKeys.select_dc_flag]:
        dc = np.array([np.zeros_like(proj[0])])
    logging.info(f"Loading done!")
    logging.info(f"Memory usage: {retrieve_memory_usage()}")

    # copy of raw data
    raw_proj = copy.deepcopy(proj)

    # crop region
    [left, right, top, bottom] = config_dict[BatchJsonKeys.crop_region]
    crop_region = [left, right, top, bottom]
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
    logging.info(f" {np.shape(proj)= }")
    logging.info(f" {np.shape(ob)= }")
    logging.info(f" {np.shape(dc)= }")
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

    return ## DEBUGGING

    # tilt_calculation
    tilt_value = config_dict[BatchJsonKeys.tilt_value]
    if tilt_value != 0.0:
        logging.info(f"Applying tilt {tilt_value} degrees ...")
        proj = diagnostics_tilt.apply_tilt_correction(arrays=proj, tilt=tilt_value)
        logging.info(f"Applying tilt done!")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f"Tilt correction skipped (tilt value is {tilt_value})!")

    # filtering #2
    if config_dict[BatchJsonKeys.remove_negative_values_flag]:
        logging.info(f"Removing negative values ...")
        proj[proj < 0] = 0
        logging.info(f"Removing negative values done!")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f"Removing negative values skipped!")

    # ring removal
    logging.info(f"Ring removal:")
    if config_dict[BatchJsonKeys.bm3d_flag]:
        logging.info(f" - BM3D processing ...")
        pass
        logging.info(f" - BM3D done!")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f" - BM3D skipped")

    if config_dict[BatchJsonKeys.tomopy_v0_flag]:
        logging.info(f" - Removing negative values ...")
        proj = tomopy.remove_all_stripe(proj, ncore=number_of_cores)
        logging.info(f" - Removing negative values ...")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f" - Removing negative values skipped")

    if config_dict[BatchJsonKeys.ketcham_flag]:
        logging.info(f" - Ketcham processing...")
        proj = remove_ring_artifact(arrays=proj,
                                    kernel_size=5,
                                    max_workers=number_of_cores)
        print(" strikes removal done!")
        logging.info(f" - Ketcham done!...")
        logging.info(f"Memory usage: {retrieve_memory_usage()}")
    else:
        logging.info(f" - Ketcham skipped")

    # rotation center
    rot_center = find_rotation_center(arrays=proj,
                                      angles=rot_angles,
                                      num_pairs=-1,
                                      in_degrees=True,
                                      atol_deg=mean_delta_angle,
                                      )

    # ranges of slices to recontruct
    [top, bottom] = config_dict[BatchJsonKeys.range_slices_to_reconstruct]
    logging.info(f"Ranges of slices to reconstruct: from {top =} to {bottom =}")

    # laminography parameters
    angle = config_dict[BatchJsonKeys.angle]
    list_gpu_index = config_dict[BatchJsonKeys.list_gpus]
    num_iter = config_dict[BatchJsonKeys.num_iterations]
    mrf_p = config_dict[BatchJsonKeys.mrf_p]
    mrf_sigma = config_dict[BatchJsonKeys.mrf_sigma]
    stop_threshold = config_dict[BatchJsonKeys.stop_threshold]
    verbose = config_dict[BatchJsonKeys.verbose]

    logging.info(f"Laminography parameters:")
    logging.info(f"\t{angle =}")
    logging.info(f"\t{list_gpu_index =}")
    logging.info(f"\t{num_iter =}")
    logging.info(f"\t {mrf_p =}")
    logging.info(f"\t{mrf_sigma =}")
    logging.info(f"\t{stop_threshold =}")
    logging.info(f"\t{verbose =}")

    # prepare command to run the laminography part

    nbr_angles, nbr_row, nbr_col = np.shape(proj)

    proj = proj.swapaxes(0, 1)
    proj = proj[top: bottom, :, :]

    raw_proj = crop(arrays=raw_proj,
                    crop_limit=crop_region)
    raw_proj = raw_proj.swapaxes(0, 1)
    raw_proj = raw_proj[top: bottom, :, :]

    rec_params = {}
    rec_params[BatchJsonKeys.num_iterations] = num_iter
    rec_params[BatchJsonKeys.list_gpus] = get_gpu_index(list_gpu_index)
    rec_params[BatchJsonKeys.mrf_p] = mrf_p
    rec_params[BatchJsonKeys.mrf_sigma] = mrf_sigma
    rec_params[BatchJsonKeys.verbose] = verbose
    rec_params[BatchJsonKeys.debug] = False
    rec_params[BatchJsonKeys.stop_threshold] = stop_threshold
    rec_params[BatchJsonKeys.filt_cutoff] = 0.5
    rec_params[BatchJsonKeys.filt_type] = 'Ram-Lak'

    proj_params = {}
    proj_params['type'] = "par"
    proj_params['dims'] = [nbr_row, nbr_angles, nbr_col]  # row, angles, col
    proj_params['angles'] = rot_angles
    laminography_angle_deg = angle
    laminography_angle_rad = np.deg2rad(laminography_angle_deg)
    alpha = np.array([laminography_angle_rad])
    proj_params['alpha'] = alpha
    proj_params['forward_model_idx'] = 2
    proj_params['pix_x'] = 1.0
    proj_params['pix_y'] = 1.0

    vol_params = {}
    [height, _, width] = proj_params['dims']
    vol_params['vox_xy'] = 1.0
    vol_params['vox_z'] = 1.0
    vol_params['n_vox_x'] = width
    vol_params['n_vox_y'] = width
    vol_params['n_vox_z'] = height

    miscalib = {}
    rot_center_pixel = rot_center  # rotation center
    off_center_u = nbr_col/2 - rot_center_pixel
    miscalib["delta_u"] = off_center_u * 1.0  
    off_center_v = 0
    miscalib["delta_v"] = off_center_v * 1.0
    miscalib["phi"] = 0

    logging.info(f"Memory usage: {retrieve_memory_usage()}")
    logging.info(f"Launching MBIR ...")
    recon_mbir = MBIR(proj, raw_proj, proj_params, miscalib, vol_params, rec_params)
    logging.info(f"MBIR done!")
    logging.info(f"Memory usage: {retrieve_memory_usage()}")
    

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
