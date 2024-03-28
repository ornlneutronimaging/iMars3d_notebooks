from qtpy.uic import loadUi

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


IN_PROGRESS = "Calculation in progress"
DONE = "Done!"
QUEUE = "In queue"


class TiltAlgorithms:
    phase_correlation = "phase correlation"
    direct_minimization = "direct minimization"
    use_center = "use center"
    scipy_minimizer = "SciPy minimizer"
    user = "user"
