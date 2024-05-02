import ipywidgets as widgets
import numpy as np
from IPython.display import display

# from tomoORNL.reconEngine import MBIR


from __code.system import System


class LaminographyEventHandler:

    def __init__(self, parent=None):
        self.parent = parent

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
        proj_tilt_corrected = self.parent.proj_tilt_corrected
        print(f"{np.shape(proj_tilt_corrected) =}")

        # proj_params['dims'] = proj_dims
        return proj_params


    def run(self):
        rec_params = self.get_rec_params()
        proj_params = self.get_proj_params()