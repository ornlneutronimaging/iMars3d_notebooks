from qtpy.uic import loadUi
import os

NCORE = 10
DEFAULT_CROP_ROI = [319, 1855, 162, 1994]
DEFAULT_BACKROUND_ROI = [0, 250, 0, 1100]

__all__ = ['load_ui']

def load_ui(ui_filename, baseinstance):
    return loadUi(ui_filename, baseinstance=baseinstance)


class DataType:
    raw = 'raw'
    ob = 'ob'
    dc = 'dc'
    ipts = 'ipts'


default_input_folder = {DataType.raw: 'ct_scans',
                        DataType.ob: 'ob',
                        DataType.dc: 'dc'}


IN_PROGRESS = "Calculation in progress"
DONE = "Done!"
QUEUE = "In queue"


class TiltAlgorithms:
    phase_correlation = "phase correlation"
    direct_minimization = "direct minimization"
    use_center = "use center"
    scipy_minimizer = "SciPy minimizer"
    user = "user"


class TiltTestKeys:
    raw_3d = 'raw_3d'
    sinogram = 'sinogram'
    center_of_rotation = 'center_of_rotation'
    reconstructed = 'reconstructed'


class ReconstructionAlgo:
    gridrec = 'gridrec'
    imars3d = 'imars3d'
    astra = 'astra'
    svmbir = 'svmbir'


class DefaultReconstructionAlgoToUse:
    gridrec = False
    imars3d = False
    astra = True
    svmbir = False


class RecParameters:
    use_this_method = "use this method"


class GridRecParameters(RecParameters):
    ratio = 'ratio'
    pad = 'pad'
    filter = 'filter'


class Imars3dParameters(RecParameters):
    pass


class AstraParameters(RecParameters):
    cpu_or_gpu = 'cpu or gpu'
    imars3d_or_astra = "imars3d or astra"
    algorithm = 'algorithm'
    ratio = 'ratio'
    nbr_iter = 'nbr iterations'
    filter = 'filter'


class SvmbirParameters(RecParameters):
    signal_to_noise = 'signal to noise ratio'
    p = 'P'
    t = 'T'
    sharpness = 'sharpness'
    max_iterations = 'max iterations'
    weight_type = 'weight type'
    verbose = 'verbose'
    temp_disk = 'temp disk'
