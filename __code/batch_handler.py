from __code.parent import Parent
from __code import DataType


class BatchHandler(Parent):

    def create_config_file(self):
        
        # input files
        list_raw_files = self.parent.input_files[DataType.raw]
        list_ob_files = self.parent.input_files[DataType.ob]
        list_dc_files = self.parent.input_files[DataType.dc]

        # crop region
        crop_left, crop_right, crop_top, crop_bottom = list(self.parent.cropping.result)

        ## filter #1
        # gamma filtering flag
        gamma_filtering_flag = self.parent.gamma_filtering_ui.value

        # beam fluctuation correction flag and region
        beam_fluctuation_flag = self.parent.beam_fluctuation_ui.value
        bf_left, bf_right, bf_top, bf_bottom = list(self.parent.beam_fluctuation_roi.value)

        # tilt value
        tilt_value = self.parent.tilt_options_ui.value

        ## filter #2
        # remove negative values
        remove_negative_values_flag = self.parent.remove_negative_ui.value

        # ring removal
        bm3d_flag = self.parent.ring_removal_ui.children[0].value
        tomopy_v0_flag = self.parent.ring_removal_ui.children[1].value
        ketcham_flag = self.parent.ring_removal_ui.children[2].value

        # range of slices to reconstruct
        top_slice, bottom_slice = list(self.parent.z_range_selection.result)

        

