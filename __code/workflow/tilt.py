import numpy as np
import os
import copy
import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from ipywidgets import interactive
import algotom.rec.reconstruction as rec
from scipy import ndimage

from imars3d.backend.diagnostics import tilt as diagnostics_tilt
from imars3d.backend.diagnostics.rotation import find_rotation_center

from __code import IN_PROGRESS, QUEUE, DONE
from __code import TiltAlgorithms, TiltTestKeys
from __code.parent import Parent
from __code import DataType
from __code import NCORE

from __code.tilt.direct_minimization import DirectMinimization
from __code.tilt.phase_correlation import PhaseCorrelation
from __code.tilt.use_center import UseCenter
from __code.utilities.math import convert_deg_in_rad


class Tilt(Parent):

    test_dict = {TiltTestKeys.raw_3d: None,
                 TiltTestKeys.sinogram: None,
                 TiltTestKeys.center_of_rotation: None,
                 TiltTestKeys.reconstructed: {},
                 }
    test_tilt_reconstruction = {TiltAlgorithms.phase_correlation: copy.deepcopy(test_dict),
                                TiltAlgorithms.direct_minimization: copy.deepcopy(test_dict),
                                TiltAlgorithms.use_center: copy.deepcopy(test_dict),
                                TiltAlgorithms.scipy_minimizer: copy.deepcopy(test_dict),
                                TiltAlgorithms.user: copy.deepcopy(test_dict),
                               }

    def find_0_180_degrees_files(self):
        rot_angles = self.parent.rot_angles

        # let's find where is index of the angle the closer to 180.0
        angles_minus_180 = rot_angles - 180.0
        abs_angles_minus_180 = np.abs(angles_minus_180)
        minimum_value = np.min(abs_angles_minus_180)

        index_0_degree = 0
        index_180_degree = np.where(minimum_value == abs_angles_minus_180)[0][0]

        rot_angles_sorted = rot_angles[:]
        rot_angles_sorted.sort()
        self.parent.mean_delta_angle = np.mean([y - x for (x, y) in zip(rot_angles_sorted[:-1],
                                                                 rot_angles_sorted[1:])])

        list_ct_files = self.parent.input_files[DataType.raw]
        short_list_cf_files = [os.path.basename(_file) for _file in list_ct_files]

        # left panel
        left_label = widgets.Label("0 degree file")
        self.parent.left_select = widgets.Select(options=short_list_cf_files,
                                          value=short_list_cf_files[index_0_degree],
                                          layout=widgets.Layout(width="500px",
                                                                height="400px"))
        left_vbox = widgets.VBox([left_label, self.parent.left_select])

        right_label = widgets.Label("180 degree file")
        self.parent.right_select = widgets.Select(options=short_list_cf_files,
                                           value=short_list_cf_files[index_180_degree],
                                           layout=widgets.Layout(width="500px",
                                                                 height="400px"))
        right_vbox = widgets.VBox([right_label, self.parent.right_select])

        hbox = widgets.HBox([left_vbox, right_vbox])
        display(hbox)

    def tilt_checkbox_handler(self, checkbox_index=1):

        list_checkboxes = [self.parent.tilt_checkbox1,
                           self.parent.tilt_checkbox2,
                           self.parent.tilt_checkbox3,
                           self.parent.tilt_checkbox4,
                           self.parent.tilt_checkbox5]
        full_list_checkboxes = list_checkboxes[:]
        list_methods = [self.tilt_checkbox1_changed,
                        self.tilt_checkbox2_changed,
                        self.tilt_checkbox3_changed,
                        self.tilt_checkbox4_changed,
                        self.tilt_checkbox5_changed]

        input_checkbox = list_checkboxes.pop(checkbox_index - 1)
        other_checkboxes = list_checkboxes

        for _check, _method in zip(full_list_checkboxes, list_methods):
            _check.unobserve(_method, names='value')

        new_value = input_checkbox.value
        if new_value is False:
            input_checkbox.value = True
            for _check, _method in zip(full_list_checkboxes, list_methods):
                _check.observe(_method, names='value')
                return

        for _check in other_checkboxes:
            _check.value = not new_value

        for _check, _method in zip(full_list_checkboxes, list_methods):
            _check.observe(_method, names='value')

    def tilt_checkbox1_changed(self, value):
        """direct minimization"""
        self.tilt_checkbox_handler(checkbox_index=1)

    def tilt_checkbox2_changed(self, value):
        """phase correlation"""
        self.tilt_checkbox_handler(checkbox_index=2)

    def tilt_checkbox3_changed(self, value):
        """use center"""
        self.tilt_checkbox_handler(checkbox_index=3)

    def tilt_checkbox4_changed(self, value):
        """scipy minimizer"""
        self.tilt_checkbox_handler(checkbox_index=4)

    def tilt_checkbox5_changed(self, value):
        """user defined"""
        self.tilt_checkbox_handler(checkbox_index=5)

    def calculate_tilt(self):

        # find out index of 0 and 180 degrees images
        self.parent.index_0_degree = self.parent.left_select.index
        self.parent.index_180_degree = self.parent.right_select.index

        # calculate the tilt using all 3 methods and let the user chose the one he wants to apply on the data
        display(HTML('<span style="font-size: 15px; color:blue">Select the tilt value you want to evaluate:</span>'))

        line1 = widgets.HBox([widgets.Checkbox(value=True,
                                               description="Direct minimization"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label(IN_PROGRESS,
                                            layout=widgets.Layout(width="200px"))
                              ])
        self.parent.tilt_checkbox1 = line1.children[0]
        self.parent.direct_minimization_value = line1.children[1]
        direct_minimization_status = line1.children[3]

        line2 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="Phase correlation"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label(QUEUE,
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.parent.tilt_checkbox2 = line2.children[0]
        self.parent.phase_correlation_value = line2.children[1]
        phase_correlation_status = line2.children[3]

        line3 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="Use center"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label(QUEUE,
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.parent.tilt_checkbox3 = line3.children[0]
        self.parent.use_center_value = line3.children[1]
        use_center_status = line3.children[3]

        # scipy minimizer
        line4 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="Scipy minimizer"),
                              widgets.Label("N/A",
                                            layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label("",
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.parent.tilt_checkbox4 = line4.children[0]
        self.parent.scipy_minimizer_value = line4.children[1]
        scipy_minimizer_status = line4.children[3]

        # user defined
        line5 = widgets.HBox([widgets.Checkbox(value=False,
                                               description="User defined"),
                              widgets.FloatText(0,
                                                layout=widgets.Layout(width="100px")),
                              widgets.Label("(deg)",
                                            layout=widgets.Layout(width="40px")),
                              widgets.Label("",
                                            layout=widgets.Layout(width="200px")),
                              ])
        self.parent.tilt_checkbox5 = line5.children[0]
        self.parent.user_value = line5.children[1]

        # self.parent.tilt_checkbox1.observe(self.tilt_checkbox1_changed, names="value")
        # self.parent.tilt_checkbox2.observe(self.tilt_checkbox2_changed, names="value")
        # self.parent.tilt_checkbox3.observe(self.tilt_checkbox3_changed, names="value")
        # self.parent.tilt_checkbox4.observe(self.tilt_checkbox4_changed, names="value")
        # self.parent.tilt_checkbox5.observe(self.tilt_checkbox5_changed, names="value")

        vertical_layout = widgets.VBox([line1, line2, line3, line4, line5])
        display(vertical_layout)

        # direct minimization
        o_direct = DirectMinimization(index_0_degree=self.parent.index_0_degree,
                                      index_180_degree=self.parent.index_180_degree,
                                      proj_mlog=self.parent.proj_mlog)
        tilt_value1 = o_direct.compute()
        self.parent.direct_minimization_value.value = f"{tilt_value1:.3f}"
        self.parent.dict_tilt_values[TiltAlgorithms.direct_minimization] = tilt_value1
        direct_minimization_status.value = DONE
        phase_correlation_status.value = IN_PROGRESS

        # phase correlation
        o_phase = PhaseCorrelation(index_0_degree=self.parent.index_0_degree,
                                   index_180_degree=self.parent.index_180_degree,
                                   proj_mlog=self.parent.proj_mlog)
        tilt_value2 = o_phase.compute()
        self.parent.phase_correlation_value.value = f"{tilt_value2:.3f}"
        self.parent.dict_tilt_values[TiltAlgorithms.phase_correlation] = tilt_value2
        phase_correlation_status.value = DONE
        use_center_status.value = IN_PROGRESS

        # use center
        o_center = UseCenter(index_0_degree=self.parent.index_0_degree,
                             index_180_degree=self.parent.index_180_degree,
                             proj_mlog=self.parent.proj_mlog)
        tilt_value3 = o_center.compute()
        self.parent.use_center_value.value = f"{tilt_value3:.3f}"
        self.parent.dict_tilt_values[TiltAlgorithms.use_center] = tilt_value3
        use_center_status.value = DONE
        scipy_minimizer_status.value = IN_PROGRESS

        # scipy minimizer
        tilt_object = diagnostics_tilt.calculate_tilt(image0=self.parent.proj_mlog[self.parent.index_0_degree],
                                                      image180=self.parent.proj_mlog[self.parent.index_180_degree])
        tilt_value4 = tilt_object.x
        self.parent.scipy_minimizer_value.value = f"{tilt_value4:.3f}"
        self.parent.dict_tilt_values[TiltAlgorithms.scipy_minimizer] = tilt_value4
        scipy_minimizer_status.value = DONE

        # user defined
        self.parent.dict_tilt_values[TiltAlgorithms.user] = self.parent.user_value.value

    def apply_tilt(self):
        tilt_value = self.get_tilt_value_selected()
        print(f"Applying tilt correction using {tilt_value:.3f} ...")
        self.parent.proj_tilt_corrected = diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                              tilt=tilt_value)

    def display_tilt(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, num="Tilt Correction", figsize=(10, 10))
        fig.suptitle("Tilt applied to data")
        index_0_image = self.parent.proj_tilt_corrected[self.parent.index_0_degree]
        index_180_image_flipped = np.fliplr(self.parent.proj_tilt_corrected[self.parent.index_180_degree])
        self.parent.overlap_image = np.add(index_0_image, index_180_image_flipped) / 2.
        fig0 = ax.imshow(self.parent.overlap_image)
        plt.colorbar(fig0, ax=ax)

    def get_tilt_value_selected(self):
        if self.parent.tilt_checkbox1.value:
            return self.parent.dict_tilt_values[TiltAlgorithms.direct_minimization]
        elif self.parent.tilt_checkbox2.value:
            return self.parent.dict_tilt_values[TiltAlgorithms.phase_correlation]
        elif self.parent.tilt_checkbox3.value:
            return self.parent.dict_tilt_values[TiltAlgorithms.use_center]
        elif self.parent.tilt_checkbox4.value:
            return self.parent.dict_tilt_values[TiltAlgorithms.scipy_minimizer]
        else:
            return self.parent.user_value.value

    def test_slices_selection(self):

        if self.parent.test_tilt_slices[0] is None:
            height, width = np.shape(self.parent.proj_norm_min)
            slice1 = 50
            slice2 = height - 50
        else:
            slice1, slice2 = self.parent.test_tilt_slices

        def plot_slices_to_reconstruct(slice1, slice2):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            ax.imshow(self.parent.proj_norm_min)

            ax.axhline(slice1, color='red', linestyle='--')
            ax.axhline(slice2, color='red', linestyle='--')

            return slice1, slice2

        nbr_angles, height, width = np.shape(self.parent.proj_mlog)

        self.reconstruct_slices = interactive(plot_slices_to_reconstruct,
                                              slice1=widgets.IntSlider(min=0,
                                                                       max=height - 1,
                                                                       value=slice1,
                                                                       continous_update=True),
                                              slice2=widgets.IntSlider(min=0,
                                                                       max=height - 1,
                                                                       value=slice2,
                                                                       continous_update=True),
                                              )
        display(self.reconstruct_slices)

    def apply_tilt_using_selected_algorithms(self):

        how_many_steps = len(self.parent.test_tilt_reconstruction)
        progress_bar = widgets.IntProgress(value=0,
                                           min=0,
                                           max=how_many_steps,
                                           description="Progress:",
                                           style={'bar-color': 'green'},
                                           )
        display(progress_bar)

        list_options = []
        if self.parent.tilt_checkbox1.value:
            list_options.append(TiltAlgorithms.direct_minimization)
            print(f"Running {TiltAlgorithms.direct_minimization} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.direct_minimization]
            self.test_tilt_reconstruction[TiltAlgorithms.direct_minimization][TiltTestKeys.raw_3d] = \
                (diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                        tilt=tilt_value))
            print("Done!")

        progress_bar.value += 1

        if self.parent.tilt_checkbox2.value:
            list_options.append(TiltAlgorithms.phase_correlation)
            print(f"Running {TiltAlgorithms.phase_correlation} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.phase_correlation]
            self.test_tilt_reconstruction[TiltAlgorithms.phase_correlation][TiltTestKeys.raw_3d] = \
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value)
            print(f"Done!")
        progress_bar.value += 1

        if self.parent.tilt_checkbox3.value:
            list_options.append(TiltAlgorithms.use_center)
            print(f"Running {TiltAlgorithms.use_center} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.use_center]
            self.test_tilt_reconstruction[TiltAlgorithms.use_center][TiltTestKeys.raw_3d] = \
            diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value)
            print(f"Done!")
        progress_bar.value += 1

        if self.parent.tilt_checkbox4.value:
            list_options.append(TiltAlgorithms.scipy_minimizer)
            print(f"Running {TiltAlgorithms.scipy_minimizer} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.scipy_minimizer]
            self.test_tilt_reconstruction[TiltAlgorithms.scipy_minimizer][TiltTestKeys.raw_3d] = (
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value))
            print(f"Done!")
        progress_bar.value += 1

        if self.parent.tilt_checkbox5.value:
            list_options.append(TiltAlgorithms.user)
            print(f"Running {TiltAlgorithms.user} ...", end=" ")
            tilt_value = self.parent.user_value.value
            self.test_tilt_reconstruction[TiltAlgorithms.user][TiltTestKeys.raw_3d] = (
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value))
            print(f"Done!")
        progress_bar.value += 1

        progress_bar.close()
        self.list_options = list_options

    def perform_reconstruction_on_selected_data_sets(self):

        slices_indexes = self.reconstruct_slices.result

        # angles in rad
        self.parent.rot_angles_rad = convert_deg_in_rad(self.parent.rot_angles)

        for key in self.test_tilt_reconstruction.keys():

            value = self.test_tilt_reconstruction[key][TiltTestKeys.raw_3d]
            if value is None:
                continue

            # convert each array (after tilt applied to it) to sinograms
            sinogram = np.moveaxis(value, 1, 0)
            self.test_tilt_reconstruction[key][TiltTestKeys.sinogram] = sinogram

            # calculate for each the center of rotation
            rot_center = find_rotation_center(arrays=value,
                                              angles=self.parent.rot_angles,
                                              num_pairs=-1,
                                              in_degrees=True,
                                              atol_deg=self.parent.mean_delta_angle,
                                              )
            self.test_tilt_reconstruction[key][TiltTestKeys.center_of_rotation] = rot_center

            # reconstruct with only selected slices
            for _slice_index in slices_indexes:
                _rec_img = rec.gridrec_reconstruction(sinogram[_slice_index],
                                                      rot_center[0],
                                                      angles=self.parent.rot_angles_rad,
                                                      apply_log=False,
                                                      ratio=1.0,
                                                      filter_name='shepp',
                                                      pad=100,
                                                      ncore=NCORE)
                self.test_tilt_reconstruction[key][TiltTestKeys.reconstructed][_slice_index] = _rec_img

    def display_test_results(self):

        if len(self.list_options) == 0:
            return

        slices_indexes = self.reconstruct_slices.result
        min_value = 10
        max_value = -10
        for _option in self.list_options:
            _min = np.min(self.test_tilt_reconstruction[self.list_options[0]][
                              TiltTestKeys.reconstructed][slices_indexes[0]])
            _max = np.max(self.test_tilt_reconstruction[self.list_options[0]][
                              TiltTestKeys.reconstructed][slices_indexes[0]])
            min_value = _min if _min < min_value else min_value
            max_value = _max if _max > max_value else max_value

            _min = np.min(self.test_tilt_reconstruction[self.list_options[0]][
                              TiltTestKeys.reconstructed][slices_indexes[1]])
            _max = np.max(self.test_tilt_reconstruction[self.list_options[0]][
                               TiltTestKeys.reconstructed][slices_indexes[1]])
            min_value = _min if _min < min_value else min_value
            max_value = _max if _max > max_value else max_value

        if len(self.list_options) > 1:
            disable_button = False
        else:
            disable_button = True

        height, width = np.shape(self.test_tilt_reconstruction[self.list_options[0]][
                    TiltTestKeys.reconstructed][slices_indexes[0]])

        init_col_value = np.floor(width/2)
        init_row_value = np.floor(height/2)

        def plot_comparisons(algo_selected, color_range, col, row, zoom_x, zoom_y):

            slice1 = self.reconstruct_slices.result[0]
            slice2 = self.reconstruct_slices.result[1]

            from_x, to_x = zoom_x
            from_y, to_y = zoom_y

            coeff_zoom_x = width / (to_x - from_x)
            coeff_zoom_y = height / (to_y - from_y)

            image_slice1 = self.test_tilt_reconstruction[algo_selected][
                    TiltTestKeys.reconstructed][slice1][from_y: to_y, from_x: to_x]
            image_slice2 = self.test_tilt_reconstruction[algo_selected][
                    TiltTestKeys.reconstructed][slice2][from_y: to_y, from_x: to_x]

            image_slice1 = ndimage.zoom(image_slice1, (coeff_zoom_y, coeff_zoom_x))
            image_slice2 = ndimage.zoom(image_slice2, (coeff_zoom_y, coeff_zoom_x))

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

            fig.suptitle(f"Tilt: {self.parent.dict_tilt_values[algo_selected]} deg")

            # top view
            ax[0][0].imshow(
                image_slice1,
                vmin=color_range[0],
                vmax=color_range[1])
            ax[0][0].set_title(f"Slice {slice1}")
            ax[0][0].axhline(row, color='blue')
            ax[0][0].axvline(col, color='blue')

            ax[1][0].imshow(
                image_slice2,
                vmin=color_range[0],
                vmax=color_range[1])
            ax[1][0].set_title(f"Slice {slice2}")
            ax[1][0].axhline(row, color='red')
            ax[1][0].axvline(col, color='red')

            # horizontal profile
            horizontal_profile_slice1 = image_slice1[row, :]
            horizontal_profile_slice2 = image_slice2[row, :]
            ax[0][1].plot(horizontal_profile_slice1, color='blue')
            ax[0][1].plot(horizontal_profile_slice2, color='red')
            ax[0][1].set_title("Horizontal profiles")

            # vertical profile
            vertical_profile_slice1 = image_slice1[:, col]
            vertical_profile_slice2 = image_slice2[:, col]
            ax[1][1].plot(vertical_profile_slice1, color='blue')
            ax[1][1].plot(vertical_profile_slice2, color='red')
            ax[1][1].set_title("Vertical profiles")

            return algo_selected

        self.test_tilt = interactive(plot_comparisons,
                                algo_selected=widgets.ToggleButtons(options=self.list_options,
                                                                    description='Algorithm:',
                                                                    disabled=disable_button),
                                color_range=widgets.FloatRangeSlider(value=[min_value, max_value],
                                                                     min=min_value,
                                                                     max=max_value,
                                                                     step=0.00001,
                                                                     ),
                                col=widgets.IntSlider(value=init_col_value,
                                                      min=0,
                                                      max=width-1),
                                row=widgets.IntSlider(value=init_row_value,
                                                      min=0,
                                                      max=height-1),
                                zoom_x=widgets.IntRangeSlider(value=[0, width-1],
                                                              min=0,
                                                              max=width-1,
                                                              continuous_update=False),
                                zoom_y=widgets.IntRangeSlider(value=[0, height-1],
                                                              min=0,
                                                              max=height-1,
                                                              continuous_update=False),
                                )
        display(self.test_tilt)

    def display_batch_options(self):
        tilt_options_ui = widgets.VBox([
            widgets.Label("Tilt value (degrees)",
                          layout=widgets.Layout(width='200px'),
                          ),
            widgets.FloatSlider(min=-90,
                                max=90,
                                value=0)
        ])
        display(tilt_options_ui)
        