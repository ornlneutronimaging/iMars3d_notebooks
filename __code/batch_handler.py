import os
import logging
from IPython.core.display import HTML
from IPython.display import display

from __code.parent import Parent
from __code import NCORE, STEP_SIZE
from __code import DataType, BatchJsonKeys
from __code.laminography_event_handler import LaminographyEventHandler
from __code.utilities.time import get_current_time_in_special_file_name_format
from __code.utilities.json import save_json


class BatchHandler(Parent):

    def create_config_file(self):
        
        # general settings
        number_of_cores = NCORE

        # input files
        list_raw_files = self.parent.input_files[DataType.raw]
        list_ob_files = self.parent.input_files[DataType.ob]
        list_dc_files = self.parent.input_files[DataType.dc]
        select_dc_flag = self.parent.select_dc_flag

        # crop region (left, right, top, bottom)
        crop_region = list(self.parent.cropping.result)

        ## filter #1
        # gamma filtering flag
        gamma_filtering_flag = self.parent.gamma_filtering_ui.value

        # beam fluctuation correction flag and region (left, right, top, bottom)
        beam_fluctuation_flag = self.parent.beam_fluctuation_ui.value
        beam_fluctuation_region = list(self.parent.beam_fluctuation_roi.result)

        # tilt value
        tilt_value = self.parent.tilt_option_ui.value

        ## filter #2
        # remove negative values
        remove_negative_values_flag = self.parent.remove_negative_ui.value

        # ring removal
        bm3d_flag = self.parent.ring_removal_ui.children[0].value
        tomopy_v0_flag = self.parent.ring_removal_ui.children[1].value
        ketcham_flag = self.parent.ring_removal_ui.children[2].value

        # range of slices to reconstruct
        range_slices_to_reconstruct = list(self.parent.z_range_selection.result)

        # laminography parameters
        ui_laminography_dict = self.parent.laminography_settings_ui
        angle = ui_laminography_dict[BatchJsonKeys.angle].value
        list_gpu_index = LaminographyEventHandler.get_gpu_index(ui_laminography_dict[BatchJsonKeys.list_gpus])
        num_iter = ui_laminography_dict[BatchJsonKeys.num_iterations].value
        mrf_p = ui_laminography_dict[BatchJsonKeys.mrf_p].value
        mrf_sigma = ui_laminography_dict[BatchJsonKeys.mrf_sigma].value
        stop_threhsold = ui_laminography_dict[BatchJsonKeys.stop_threshold].value
        verbose = ui_laminography_dict[BatchJsonKeys.verbose].value
        laminography_dict = {BatchJsonKeys.angle: angle,
                            BatchJsonKeys.list_gpus: list_gpu_index,
                            BatchJsonKeys.num_iterations: num_iter,
                            BatchJsonKeys.mrf_p: mrf_p,
                            BatchJsonKeys.mrf_sigma: mrf_sigma,
                            BatchJsonKeys.stop_threshold: stop_threhsold,
                            BatchJsonKeys.verbose: verbose}

        # output folder
        output_folder = self.parent.output_folder

        # create names of output and config file
        _current_time = get_current_time_in_special_file_name_format()
        base_folder_name = self.parent.raw_folder_name
        json_file_name = os.path.join(os.path.expanduser("~"), 
                                      f"laminography_{base_folder_name}_{_current_time}.json")

        log_file_name = os.path.join(os.path.expanduser("~"),
                                     f"laminography_{base_folder_name}_{_current_time}.log")

        logging.basicConfig(filename=log_file_name,
                            filemode='a',
                            format='[%(levelname)s] - %(asctime)s - %(message)s',
                            level=logging.INFO)
        logging.info("*** Starting a new process ***")

        # create json dictionary
        json_dictionary = {BatchJsonKeys.list_raw_files: list_raw_files,
                           BatchJsonKeys.list_ob_files: list_ob_files,
                           BatchJsonKeys.list_dc_files: list_dc_files,
                           BatchJsonKeys.select_dc_flag: select_dc_flag,
                           BatchJsonKeys.crop_region: crop_region,
                           BatchJsonKeys.gamma_filtering_flag: gamma_filtering_flag,
                           BatchJsonKeys.beam_fluctuation_flag:beam_fluctuation_flag,
                           BatchJsonKeys.beam_fluctuation_region: beam_fluctuation_region,
                           BatchJsonKeys.tilt_value: tilt_value,
                           BatchJsonKeys.remove_negative_values_flag: remove_negative_values_flag,
                           BatchJsonKeys.bm3d_flag: bm3d_flag,
                           BatchJsonKeys.tomopy_v0_flag: tomopy_v0_flag,
                           BatchJsonKeys.ketcham_flag: ketcham_flag,
                           BatchJsonKeys.range_slices_to_reconstruct: range_slices_to_reconstruct,
                           BatchJsonKeys.laminography_dict: laminography_dict,
                           BatchJsonKeys.output_folder: output_folder,
                           BatchJsonKeys.log_file_name: log_file_name,
                           BatchJsonKeys.number_of_cores: number_of_cores,
                           BatchJsonKeys.step_size: STEP_SIZE,
                           }

        save_json(json_file_name=json_file_name,
                  json_dictionary=json_dictionary)
        logging.info(f"json file created: {json_file_name}")

        display(HTML(f"<h2>Config file:</h2> {json_file_name}"))
        display(HTML(f"<h2>LOG file:</h2> {log_file_name}"))
