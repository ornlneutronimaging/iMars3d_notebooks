from qtpy.uic import loadUi

NCORE = 10

__all__ = ['load_ui']


def load_ui(ui_filename, baseinstance):
    return loadUi(ui_filename, baseinstance=baseinstance)


