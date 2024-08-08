import ipywidgets as widgets
import numpy as np
from IPython.display import display
from ipywidgets import interactive
import matplotlib.pyplot as plt
import time

from tomoORNL.reconEngine import MBIR
from imars3d.backend.morph.crop import crop
from imars3d.backend.dataio.data import save_data

from __code.system import System
from __code.utilities.files import save_json
from __code.utilities.time import convert_time_s_in_time_hr_mn_s
from __code import BatchJsonKeys
from __code.utilities.system import print_memory_usage, delete_array


class LaminographyEventHandler:

    huber_t = 5
    huber_delta = 0.1
    sigma = 1
    reject_frac = 0.1
    debug = False

    det_x = 1.0
    det_y = 1.0

    forward_model_idx = 2

    vox_xy = 1.0
    vox_z = 1.0
    n_vox_x = 256
    n_vox_y = 256
    n_vox_z = 256

    off_center_u = 0  # Center of rotation offset in units of pixels
    off_center_v = 0

    nbr_angles = 0
    nbr_row = 0
    nbr_col = 0

    _proj_mlog = None

    def __init__(self, parent=None, batch_mode=False):
        self.parent = parent

        self.z_top, self.z_bottom = list(self.parent.z_range_selection.result)
        if batch_mode:
            proj_mlog = self.parent.proj_raw
        else:
            proj_mlog = self.parent.proj_mlog
        self._proj_mlog = proj_mlog[self.z_top: self.z_bottom, :, :]

        [self.nbr_angles, self.nbr_row, self.nbr_col] = np.shape(self._proj_mlog)

    def set_settings(self):
        self.laminography_angle_ui = widgets.FloatSlider(min=0, max=90, value=20, step=0.01)
        tab1 = widgets.HBox([self.laminography_angle_ui,
                             widgets.Label(value=u"\u00b0")
                             ])

        nbr_gpu = System.get_number_of_gpu()
        gpu_layout = widgets.HBox()
        self.children_gpus = [widgets.Label(value="GPUs:",
                                            layout=widgets.Layout(width="20%"))]
        for _index_gpu in np.arange(nbr_gpu):
            self.children_gpus.append(widgets.Checkbox(value=True,
                                                       description=f"#{_index_gpu + 1}"))
        gpu_layout.children = self.children_gpus

        self.num_iter_ui = widgets.IntSlider(min=1, max=500, value=200, description="Num. itera.")
        self.mrf_p_ui = widgets.FloatSlider(min=0, max=5, value=1.2, description="MRF P")
        self.mrf_sigma_ui = widgets.FloatSlider(min=0, max=5, value=0.5, description="MRF Sigma")
        self.stop_threshold_ui = widgets.BoundedFloatText(min=0, max=1, value=0.001, step=0.001,
                                                          description="Stop thresh.")
        self.verbose_ui = widgets.Checkbox(value=False, description="Verbose")
        tab2 = widgets.VBox([self.num_iter_ui,
                             gpu_layout,
                             self.mrf_p_ui,
                             self.mrf_sigma_ui,
                             self.stop_threshold_ui,
                             self.verbose_ui,
                             ])

        tab = widgets.Tab([tab1, tab2])
        tab.titles = ["Laminography angle", "Advanced settings"]
        display(tab)

        # saving widgets for batch mode
        self.parent.laminography_settings_ui = {BatchJsonKeys.angle: self.laminography_angle_ui,
                                                BatchJsonKeys.list_gpus: self.children_gpus,
                                                BatchJsonKeys.num_iterations: self.num_iter_ui,
                                                BatchJsonKeys.mrf_p: self.mrf_p_ui,
                                                BatchJsonKeys.mrf_sigma: self.mrf_sigma_ui,
                                                BatchJsonKeys.stop_threshold: self.stop_threshold_ui,
                                                BatchJsonKeys.verbose: self.verbose_ui}

    @staticmethod
    def get_gpu_index(children_gpus_ui):
        gpu_index = []
        for _index, _child in enumerate(children_gpus_ui):
            if _child.value:
                gpu_index.append(_index)
        return gpu_index

    def get_rec_params(self):
        rec_params = {}
        rec_params[BatchJsonKeys.num_iterations] = self.num_iter_ui.value
        rec_params[BatchJsonKeys.list_gpus] = LaminographyEventHandler.get_gpu_index(self.children_gpus[1:])
        rec_params[BatchJsonKeys.mrf_p] = self.mrf_p_ui.value
        rec_params[BatchJsonKeys.mrf_sigma] = self.mrf_sigma_ui.value
        # rec_params['huber_T'] = self.huber_t
        # rec_params['huber_delta'] = self.huber_delta
        # rec_params['sigma'] = self.sigma
        # rec_params['reject_frac'] = self.reject_frac
        rec_params[BatchJsonKeys.verbose] = self.verbose_ui.value
        rec_params[BatchJsonKeys.debug] = self.debug
        rec_params[BatchJsonKeys.stop_threshold] = self.stop_threshold_ui.value
        rec_params[BatchJsonKeys.filt_cutoff] = 0.5
        rec_params[BatchJsonKeys.filt_type] = 'Ram-Lak'

        return rec_params

    def get_proj_params(self):
        proj_params = {}

        proj_params['type'] = "par"

        # row, angles, col
        # proj_dim = np.array([self.nbr_row, self.nbr_angles, self.nbr_col])
        proj_dim = np.shape(self._proj_data)
        proj_params['dims'] = proj_dim

        # angles
        angles = self.parent.rot_angles_rad
        proj_params['angles'] = angles

        # alpha
        laminography_angle_deg = self.laminography_angle_ui.value
        laminography_angle_rad = np.deg2rad(laminography_angle_deg)
        alpha = np.array([laminography_angle_rad])
        proj_params['alpha'] = alpha

        proj_params['forward_model_idx'] = self.forward_model_idx

        proj_params['pix_x'] = self.det_x
        proj_params['pix_y'] = self.det_y

        return proj_params

    def get_vol_params(self):
        vol_params = {}

        [height, _, width] = np.shape(self._proj_data)

        vol_params['vox_xy'] = self.vox_xy
        vol_params['vox_z'] = self.vox_z
        vol_params['n_vox_x'] = width
        vol_params['n_vox_y'] = width
        vol_params['n_vox_z'] = height

        return vol_params

    def get_miscalib(self):
        miscalib = {}

        # rotation center
        rot_center_pixel = self.parent.rot_center[0]
        nbr_col = self.nbr_col
        off_center_u = nbr_col/2 - rot_center_pixel
        miscalib["delta_u"] = off_center_u * self.det_x
       
        off_center_v = 0
        miscalib["delta_v"] = off_center_v * self.det_y

        # tilt_value
        # tilt_algo_selected = self.parent.tilt_algo_selected_finally
        # tilt_value_deg = self.parent.dict_tilt_values[tilt_algo_selected]
        # tilt_value_rad = np.deg2rad(tilt_value_deg)
        miscalib["phi"] = 0

        return miscalib

    def run(self):

        # change the axis order from [angles, row, columns] to [row, angles, columns]
        # proj_data = np.moveaxis(self.parent.proj_tilt_corrected, 1, 0)
        proj_data = self.parent.proj_tilt_corrected.swapaxes(0, 1)
        self._proj_data = proj_data[self.z_top: self.z_bottom, :, :]

        # crop raw data (just it was done for proj_data
        crop_region = self.parent.crop_region
        weight_data_raw = crop(arrays=self.parent.untouched_sample_data,
                               crop_limit=crop_region)
        # weight_data = np.moveaxis(weight_data_raw, 1, 0)
        weight_data = weight_data_raw.swapaxes(0, 1)
        self._weight_data = weight_data[self.z_top: self.z_bottom, :, :]

        rec_params = self.get_rec_params()
        proj_params = self.get_proj_params()
        vol_params = self.get_vol_params()
        miscalib = self.get_miscalib()

        debug_json = {'size of data': np.shape(self._proj_data),
                      'size of weight': np.shape(self._weight_data),
                      'proj_params': proj_params,
                      'miscalib': miscalib,
                      'vol_params': vol_params,
                      'rec_params': rec_params,
                      }
        # json_file_name = '/SNS/users/j35/debug_laminography.json'
        # save_json(json_dictionary=debug_json, json_file_name=json_file_name)
        # import pprint
        # pprint.pprint(f"{debug_json =}")

        start_time = time.time()

        # print(f"{np.shape(self._proj_data)= }")
        # print(f"{np.shape(self._weight_data)= }")

        # print(f"{proj_params =}")
        # print(f"{miscalib =}")
        # print(f"{vol_params =}")
        # print(f"{rec_params =}")

        self.parent.recon_mbir = MBIR(self._proj_data, 
                                      self._weight_data, 
                                      proj_params, 
                                      miscalib, 
                                      vol_params, 
                                      rec_params)
        end_time = time.time()
        delta_time = end_time - start_time
        print(f"Laminography reconstruction ran in {convert_time_s_in_time_hr_mn_s(delta_time)}")

        self.export_data_for_debugging(data_to_export=self.parent.recon_mbir)

    def visualize(self):

        # recon_mbir = self.parent.recon_mbir.swapaxes(0, 1)
        recon_mbir = self.parent.recon_mbir
        [dim1, _, _] = np.shape(recon_mbir)

        def plot_final_volume(index):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            ax.imshow(recon_mbir[index, :, :])

        display_volume = interactive(plot_final_volume,
                                     index=widgets.IntSlider(min=0, 
                                                             max=dim1-1,
                                                             continuous_update=False))
        display(display_volume)

    def export_data_for_debugging(self, data_to_export=None):
        print("Exporting data for debugging only")
        outputbase = "/HFIR/CG1D/IPTS-23768/shared/processed_data/jean/"
        folder_name = "data_created_by_notebook"

        print(f"before swap: {np.shape(data_to_export) =}")

        # switch back to [index, y, x]
        data_to_export = data_to_export.swapaxes(1, 0)

        print(f"after swap: {np.shape(data_to_export) =}")

        save_data(data=data_to_export,
                outputbase=outputbase,
                name=folder_name)
        
        print("Done!")
