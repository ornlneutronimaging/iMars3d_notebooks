import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display

from __code.parent import Parent


class Display(Parent):

    def sinogram(self, sinogram_data):

        def plot_sinogram(index):
            fig, axis = plt.subplots(num="sinogram", figsize=(10, 10), nrows=1, ncols=1)
            axis.imshow(sinogram_data[index])
            axis.set_title(f"Sinogram at slice #{index}")

        plot_sinogram_ui = interactive(plot_sinogram,
                                       index=widgets.IntSlider(min=0,
                                                               max=len(sinogram_data),
                                                               value=0))
        display(plot_sinogram_ui)
