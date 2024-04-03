import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display
from __code.parent import Parent
import matplotlib.pyplot as plt
from IPython.core.display import HTML
import algotom.rec.reconstruction as rec
from scipy import ndimage


from __code import NCORE


class TestReconstruction(Parent):

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

        # astra
        self.astra_layout = widgets.VBox([widgets.Checkbox(value=True,
                                                      description="Use this method?"),
                                     widgets.RadioButtons(options=["CPU", "GPU"]),
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
                                                self.astra_layout,
                                                self.svmbir_layout],
                                      titles=('Gridrec', 'ASTRA', 'svMBIR'))
        accordion.selected_index = 0
        display(accordion)

    def running_reconstruction_test(self):
        print("running reconstruction test")