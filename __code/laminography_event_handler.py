import ipywidgets as widgets
import numpy as np
from IPython.display import display

from tomoORNL.reconEngine import MBIR
from imars3d.backend.morph.crop import crop


from __code.system import System


class LaminographyEventHandler:

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

    def __init__(self, parent=None):
        self.parent = parent
        [self.nbr_angles, self.nbr_row, self.nbr_col] = np.shape(self.parent.proj_mlog)

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

        self.huber_t_ui = widgets.FloatText(value=5, description="Huber T")
        self.huber_delta_ui = widgets.FloatText(value=0.1, description="Huber delta")
        self.sigma_ui = widgets.FloatText(value=1, description="Sigma")
        self.reject_frac_ui = widgets.FloatText(value=0.1, description="Reject. frac.")
        self.debug_ui = widgets.Checkbox(value=False, description="Debug")
        tab3 = widgets.VBox([self.huber_t_ui,
                             self.huber_delta_ui,
                             self.sigma_ui,
                             self.reject_frac_ui,
                             self.debug_ui])

        tab = widgets.Tab([tab1, tab2, tab3])
        tab.titles = ["Laminography angle", "Settings", "Advanced settings"]
        display(tab)

    def get_gpu_index(self):
        children_gpus_ui = self.children_gpus[1:]
        gpu_index = []
        for _index, _child in enumerate(children_gpus_ui):
            if _child.value:
                gpu_index.append(_index)
        return gpu_index

    def get_rec_params(self):
        rec_params = {}
        rec_params['num_iter'] = self.num_iter_ui.value
        rec_params['gpu_index'] = self.get_gpu_index()
        rec_params['MRF_P'] = self.mrf_p_ui.value
        rec_params['MRF_SIGMA'] = self.mrf_sigma_ui.value
        rec_params['huber_T'] = self.huber_t_ui.value
        rec_params['huber_delta'] = self.huber_delta_ui.value
        rec_params['sigma'] = self.sigma_ui.value
        rec_params['reject_frac'] = self.reject_frac_ui.value
        rec_params['verbose'] = self.verbose_ui.value
        rec_params['debug'] = self.debug_ui.value
        rec_params['stop_thresh'] = self.stop_threshold_ui.value

    def get_proj_params(self):
        proj_params = {}

        proj_params['type'] = "par"

        # row, angles, col
        proj_dim = np.array([self.nbr_row, self.nbr_angles, self.nbr_col])
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

        vol_params['vox_xy'] = self.vox_xy
        vol_params['vox_z'] = self.vox_z
        vol_params['n_vox_x'] = self.n_vox_x
        vol_params['n_vox_y'] = self.n_vox_y
        vol_params['n_vox_z'] = self.n_vox_z

        return vol_params

    def miscalib(self):
        miscalib = {}

        # rotation center
        rot_center_pixel = self.parent.rot_center[0]
        nbr_col = self.nbr_col
        off_center_u = rot_center_pixel - nbr_col/2
        miscalib["delta_u"] = off_center_u * self.det_x

        off_center_v = 0
        miscalib["delta_v"] = off_center_v * self.det_y

        # tilt_value
        tilt_algo_selected = self.parent.tilt_algo_selected_finally
        tilt_value_deg = self.parent.dict_tilt_values[tilt_algo_selected]
        tilt_value_rad = np.deg2rad(tilt_value_deg)
        miscalib["phi"] = tilt_value_rad

        return miscalib

    def run(self):

        # change the axis order from [angles, row, columns] to [row, angles, columns]
        proj_data = np.moveaxis(self.parent.proj_mlog, 1, 0)

        # raw data
        weight_data_raw = np.moveaxis(self.parent.untouched_sample_data, 1, 0)
        # crop raw data (just it was done for proj_data
        crop_region = self.parent.crop_region
        weight_data = crop(arrays=weight_data_raw,
                           crop_limit=crop_region)

        rec_params = self.get_rec_params()
        proj_params = self.get_proj_params()
        vol_params = self.get_vol_params()
        miscalib = self.get_miscalib()

        self.parent.recon_mbir = MBIR(proj_data, weight_data, proj_params, miscalib, vol_params, rec_params)
