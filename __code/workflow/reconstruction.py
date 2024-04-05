import timeit

import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
from __code.parent import Parent
import matplotlib.pyplot as plt
from IPython.core.display import HTML
import algotom.rec.reconstruction as rec
from scipy import ndimage

from imars3d.backend.reconstruction import recon

from __code import NCORE
from __code import ReconstructionAlgo, GridRecParameters, AstraParameters, SvmbirParameters, Imars3dParameters
from __code import TiltTestKeys


class TestReconstruction(Parent):

    test_reconstruction_dict = {}

    def define_slices_to_test_reconstruction(self):
        height, width = np.shape(self.parent.overlap_image)
        nbr_slices = 4
        step = height / (nbr_slices + 1)
        slices = [k * step for k in np.arange(1, nbr_slices + 1)]

        display(
            HTML("<span style='color:blue'><b>Position of the slices you want to test the reconstruction with:</b>" +
                 "<br></span><b>To add a new slice</b>, enter value to the right of the last slice defined"))

        def display_image_and_slices(list_slices):
            fig, axs = plt.subplots(num='Select slices to reconstruct')
            fig.set_figwidth(15)
            axs.imshow(self.parent.overlap_image)
            for _slice in list_slices:
                axs.axhline(_slice, color='red', linestyle='--')

            return list_slices

        self.display_slices = interactive(display_image_and_slices,
                                          list_slices=widgets.IntsInput(value=slices,
                                                                        min=0,
                                                                        max=height - 1))
        display(self.display_slices)

    def test_reconstruction(self):
        list_slices = self.display_slices.result
        rec_images = []
        for num, idx in enumerate(list_slices):
            rec_images.append(rec.gridrec_reconstruction(self.parent.sinogram_mlog[idx],
                                                         self.parent.rot_center[0],
                                                         angles=self.parent.rot_angles,
                                                         apply_log=False,
                                                         ratio=1.0,
                                                         filter_name='shepp',
                                                         pad=100,
                                                         ncore=NCORE))

        height, width = np.shape(rec_images[0])

        # display slices reconstructed here
        def display_slices(slice_index, zoom_x, zoom_y):
            from_x, to_x = zoom_x
            from_y, to_y = zoom_y

            coeff_zoom_x = width / (to_x - from_x)
            coeff_zoom_y = height / (to_y - from_y)

            image_slice = rec_images[slice_index]
            image_slice_crop = image_slice[from_y: to_y, from_x: to_x]
            image_slice_zoom = ndimage.zoom(image_slice_crop, (coeff_zoom_y, coeff_zoom_x))

            fig, axs = plt.subplots(num="testing reconstruction", ncols=2, nrows=1)
            fig.set_figwidth(15)

            axs[0].imshow(image_slice_zoom)
            axs[0].set_title(f"Slice {list_slices[slice_index]}.")
            axs[1].imshow(self.parent.overlap_image)
            axs[1].set_title(f"Integrated image and slice {list_slices[slice_index]} position.")

            axs[1].axhline(list_slices[slice_index], color='red', linestyle='--')

        display_test = interactive(display_slices,
                                   slice_index=widgets.IntSlider(min=0,
                                                                 max=len(list_slices) - 1,
                                                                 continuous_update=True),
                                   zoom_x=widgets.IntRangeSlider(value=[0, width - 1],
                                                                 min=0,
                                                                 max=width - 1,
                                                                 continuous_update=False),
                                   zoom_y=widgets.IntRangeSlider(value=[0, height - 1],
                                                                 min=0,
                                                                 max=height - 1,
                                                                 continuous_update=False),
                                   )
        display(display_test)

    def testing_reconstruction_algorithms(self):
        display(widgets.HTML("<font color='blue'>Define reconstruction algorithms to use and their settings:"))

        # gridrec
        self.gridrec_layout = widgets.VBox([widgets.Checkbox(value=True,
                                                        description="Use this method?"),
                                       widgets.FloatText(value=1.0,
                                                         description="Ratio:",
                                                         step=0.1),
                                       widgets.IntText(value=100,
                                                       description="Pad:"),
                                       widgets.HBox([widgets.Label("Filter:"),
                                                     widgets.Label("shepp")]),
                                       ])

        # imars3d
        self.imars3d_layout = widgets.VBox([widgets.Checkbox(value=True,
                                                      description="Use this method?"),
                                            ])

        # astra
        self.astra_layout = widgets.VBox([widgets.Checkbox(value=True,
                                                      description="Use this method?"),
                                     widgets.RadioButtons(options=["CPU", "GPU"],
                                                          disabled=True),
                                     widgets.Select(options=['FBP', 'SIRT', 'ART', 'CGLS'],
                                                    description="Algorithm"),
                                     widgets.FloatText(value=1.0,
                                                       description="Ratio:",
                                                       step=0.1),
                                     widgets.IntText(value=300,
                                                     description="Nbr iter:"),
                                     widgets.HBox([widgets.Label("Filter:"),
                                                   widgets.Label("hann")])])

        # svmbir
        self.svmbir_layout = widgets.VBox([widgets.Checkbox(value=True,
                                                       description="Use this method?"),
                                      widgets.HBox([widgets.Label(value='Signal to noise:'),
                                                    widgets.FloatText(value=30.0,
                                                                      layout=widgets.Layout(width="50px"))]),
                                      widgets.FloatSlider(value=1.2,
                                                          min=1,
                                                          max=2,
                                                          description="P"),
                                      widgets.FloatSlider(value=2.0,
                                                          min=0,
                                                          max=10,
                                                          description="T"),
                                      widgets.FloatSlider(value=0.0,
                                                          min=0,
                                                          max=100,
                                                          description="Sharpness"),
                                      widgets.HBox([widgets.Label(value='Max iterations:'),
                                                    widgets.IntText(value=100,
                                                                    layout=widgets.Layout(width="50px"))]),
                                      widgets.HBox([widgets.Label("Weight type:"),
                                                    widgets.Label("transmission")]),
                                      widgets.Checkbox(value=False,
                                                       description="Verbose:"),
                                      widgets.HBox([widgets.Label("Temp disk:"),
                                                    widgets.Label("/netdisk/y9z/svmbir_cache")])

                                      ])

        accordion = widgets.Accordion(children=[self.gridrec_layout,
                                                self.imars3d_layout,
                                                self.astra_layout,
                                                self.svmbir_layout],
                                      titles=('Gridrec', 'ASTRA', 'svMBIR'))
        accordion.selected_index = 0
        display(accordion)

    def retrieving_parameters(self):

        gridrec_layout = self.gridrec_layout
        imars3d_layout = self.imars3d_layout
        astra_layout = self.astra_layout
        svmbir_layout = self.svmbir_layout

        gridrec_dict = {GridRecParameters.use_this_method: gridrec_layout.children[0].value,
                        GridRecParameters.ratio: gridrec_layout.children[1].value,
                        GridRecParameters.pad: gridrec_layout.children[2].value,
                        GridRecParameters.filter: gridrec_layout.children[3].children[1].value}

        imars3d_dict = {Imars3dParameters.use_this_method: imars3d_layout.children[0].value}

        astra_dict = {AstraParameters.use_this_method: astra_layout.children[0].value,
                      AstraParameters.cpu_or_gpu: astra_layout.children[1].value,
                      AstraParameters.algorithm: astra_layout.children[2].value,
                      AstraParameters.ratio: astra_layout.children[3].value,
                      AstraParameters.nbr_iter: astra_layout.children[4].value,
                      AstraParameters.filter: astra_layout.children[5].children[1].value}

        svmbir_dict = {SvmbirParameters.use_this_method: svmbir_layout.children[0].value,
                       SvmbirParameters.signal_to_noise: svmbir_layout.children[1].children[1].value,
                       SvmbirParameters.p: svmbir_layout.children[2].value,
                       SvmbirParameters.t: svmbir_layout.children[3].value,
                       SvmbirParameters.sharpness: svmbir_layout.children[4].value,
                       SvmbirParameters.max_iterations: svmbir_layout.children[5].children[1].value,
                       SvmbirParameters.weight_type: svmbir_layout.children[6].children[1].value,
                       SvmbirParameters.verbose: svmbir_layout.children[7].value,
                       SvmbirParameters.temp_disk: svmbir_layout.children[8].children[1].value,
                       }

        self.test_reconstruction_dict[ReconstructionAlgo.gridrec] = gridrec_dict
        self.test_reconstruction_dict[ReconstructionAlgo.imars3d] = imars3d_dict
        self.test_reconstruction_dict[ReconstructionAlgo.astra] = astra_dict
        self.test_reconstruction_dict[ReconstructionAlgo.svmbir] = svmbir_dict

    def running_reconstruction_test(self):
        print("running reconstruction test")

        # list of slices to use to reconstruct
        list_slices = self.display_slices.result

        # sinogram
        sinogram = self.parent.sinogram_after_ring_removal

        # projection after ring removed
        proj_ring_removed = self.parent.proj_ring_removed

        # center_of_rotation
        tilt_algo_selected = self.parent.o_tilt.test_tilt.result
        rot_center = self.parent.o_tilt.test_tilt_reconstruction[tilt_algo_selected][TiltTestKeys.center_of_rotation][0]

        # list of angles
        rot_angles_rad = self.parent.rot_angles_rad
        rot_angles_deg = self.parent.rot_angles

        # gridrec
        if self.test_reconstruction_dict[ReconstructionAlgo.gridrec][GridRecParameters.use_this_method]:

            print("\t> testing reconstruction using gridrec:")
            gridrec_dict = self.test_reconstruction_dict[ReconstructionAlgo.gridrec]
            ratio = gridrec_dict[GridRecParameters.ratio]
            filter = gridrec_dict[GridRecParameters.filter]
            pad = gridrec_dict[GridRecParameters.pad]

            t_start = timeit.default_timer()
            for slice in list_slices:
                rec_img = rec.gridrec_reconstruction(sinogram[slice],
                                                     rot_center,
                                                     angles=rot_angles_rad,
                                                     apply_log=False,
                                                     ratio=ratio,
                                                     filter_name=filter,
                                                     pad=pad,
                                                     ncore=NCORE)
            t_end = timeit.default_timer()
            print(f"\t Gridrec ran in {(t_end - t_start)/60} mns")

        if self.test_reconstruction_dict[ReconstructionAlgo.astra][GridRecParameters.use_this_method]:
            print("\t> testing reconstruction using astra:")
            astra_dict = self.test_reconstruction_dict[ReconstructionAlgo.astra]

            algorithm = astra_dict[AstraParameters.algorithm]
            ratio = astra_dict[AstraParameters.ratio]
            nbr_iterations = astra_dict[AstraParameters.nbr_iter]
            filter = astra_dict[AstraParameters.filter]

            t_start = timeit.default_timer()
            if astra_dict[AstraParameters.cpu_or_gpu] == 'CPU':  # using CPU
                pass
            else:
                algorithm = f"{algorithm}_CUDA"   # using GPU

            for slice in list_slices:
                rec_img = rec.astra_reconstruction(sinogram[slice],
                                                   rot_center[0],
                                                   angles=rot_angles_rad,
                                                   apply_log=False,
                                                   method=algorithm,
                                                   ratio=ratio,
                                                   filter_name=filter,
                                                   pad=None,
                                                   num_iter=nbr_iterations,
                                                   ncore=NCORE)

            t_end = timeit.default_timer()
            print(f"\t Gridrec ran in {(t_end - t_start)/60} mns")

        if self.test_reconstruction_dict[ReconstructionAlgo.svmbir][GridRecParameters.use_this_method]:
            print("\t> testing reconstruction using svMBIR:")
