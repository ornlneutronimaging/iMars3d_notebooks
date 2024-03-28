import numpy as np
import os
import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from ipywidgets import interactive

from imars3d.backend.diagnostics import tilt as diagnostics_tilt

from __code import IN_PROGRESS, QUEUE, DONE
from __code import TiltAlgorithms
from __code.parent import Parent
from __code import DataType

from __code.tilt.direct_minimization import DirectMinimization
from __code.tilt.phase_correlation import PhaseCorrelation
from __code.tilt.use_center import UseCenter


class Tilt(Parent):

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
        display(HTML('<span style="font-size: 15px; color:blue">Select the tilt value you want to use:</span>'))

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

        self.parent.tilt_checkbox1.observe(self.tilt_checkbox1_changed, names="value")
        self.parent.tilt_checkbox2.observe(self.tilt_checkbox2_changed, names="value")
        self.parent.tilt_checkbox3.observe(self.tilt_checkbox3_changed, names="value")
        self.parent.tilt_checkbox4.observe(self.tilt_checkbox4_changed, names="value")
        self.parent.tilt_checkbox5.observe(self.tilt_checkbox5_changed, names="value")

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

    def apply_tilt_and_display(self):
        tilt_value = self.get_tilt_value_selected()
        print(f"Applying tilt correction using {tilt_value:.3f} ...")
        self.parent.proj_tilt_corrected = diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                              tilt=tilt_value)

        fig, ax = plt.subplots(nrows=1, ncols=1, num="Tilt Correction", figsize=(10, 10))

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

    def testing_algo_selected(self):

        how_many_steps = len(self.parent.test_tilt_reconstruction)
        progress_bar = widgets.IntProgress(value=0,
                                           min=0,
                                           max=how_many_steps,
                                           description="Progress",
                                           style={'bar-color': 'green'})
        display(progress_bar)

        list_options = []
        if self.parent.tilt_checkbox1.value:
            list_options.append(TiltAlgorithms.direct_minimization)
            print(f"Running {TiltAlgorithms.direct_minimization} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.direct_minimization]
            self.parent.test_tilt_reconstruction[TiltAlgorithms.direct_minimization] = (
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value))
            print("Done!")

        progress_bar.value += 1

        if self.parent.tilt_checkbox2.value:
            list_options.append(TiltAlgorithms.phase_correlation)
            print(f"Running {TiltAlgorithms.phase_correlation} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.phase_correlation]
            self.parent.test_tilt_reconstruction[TiltAlgorithms.phase_correlation] = (
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value))
            print(f"Done!")
        progress_bar.value += 1

        if self.parent.tilt_checkbox3.value:
            list_options.append(TiltAlgorithms.use_center)
            print(f"Running {TiltAlgorithms.use_center} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.use_center]
            self.parent.test_tilt_reconstruction[TiltAlgorithms.use_center] = (
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value))
            print(f"Done!")
        progress_bar.value += 1

        if self.parent.tilt_checkbox4.value:
            list_options.append(TiltAlgorithms.scipy_minimizer)
            print(f"Running {TiltAlgorithms.scipy_minimizer} ...", end=" ")
            tilt_value = self.parent.dict_tilt_values[TiltAlgorithms.scipy_minimizer]
            self.parent.test_tilt_reconstruction[TiltAlgorithms.scipy_minimizer] = (
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value))
            print(f"Done!")
        progress_bar.value += 1

        if self.parent.tilt_checkbox5.value:
            list_options.append(TiltAlgorithms.user)
            print(f"Running {TiltAlgorithms.user} ...", end=" ")
            tilt_value = self.parent.user_value.value
            self.parent.test_tilt_reconstruction[TiltAlgorithms.user] = (
                diagnostics_tilt.apply_tilt_correction(arrays=self.parent.proj_mlog,
                                                       tilt=tilt_value))
            print(f"Done!")
        progress_bar.value += 1

        progress_bar.close()
        self.list_options = list_options

    def display_results(self):

        if len(self.list_options) == 0:
            return

        slices_indexes = self.reconstruct_slices

        if len(self.list_options) > 1:

            def plot_comparisons(value):

                print(f"{value =}")
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

                slice1 = self.parent


            test_tilt = interactive(plot_comparisons,
                                    value=widgets.ToggleButtons(options=self.list_options),
                                    )
            display(test_tilt)





