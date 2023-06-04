"""heatlab __init__ module"""

from typing import Callable

import PyQt5.QtWidgets as qtw

from .assets import *
from .segmenters import *
from .utils import *
from .widgets import *
from .io import *


def run(
    sgm_list: list[Segmenter] | Segmenter | None = None,
    file_formats: dict[str, Callable] | None = None,
):
    """
    run the HeatmapLabeller application with optional custom parameters

    Parameters
    ----------
    sgm_list : list[segmenters.Segmenter] | segmenters.Segmenter | None
        by default None, the segmenters to be available since the beginning

    file_formats : dict[str, Callable] | None
        by default None, the file formats to be read. Each key of the dictionary
        must point to a function accepting a string denoting the file path
        and returning a 4D numpy array with shape
        (frame, height, width, 1 | 4) where the last dimension is 1 for
        heatmaps and greyscale images or 4 for RGBA images.
    """
    # create the app
    app = qtw.QApplication([])
    app.setStyle("Fusion")

    # generate the segmenters
    if sgm_list is None:
        ell = EllipseSegmenter(
            name="ELLIPSE",
            color=(255, 0, 0, 255),
            linewidth=1,
            fontsize=4,
        )
        tri = TriangleSegmenter(
            name="TRIANGLE",
            color=(0, 255, 0, 255),
            linewidth=1,
            fontsize=4,
        )
        sgm_list = [ell, tri]

    # generate the data reading formats
    if file_formats is None:
        file_formats = io.SUPPORTED_EXTENSIONS

    # generate the labeller
    labeller = LabellerWidget(sgm_list, **file_formats)

    # run
    labeller.show()
    app.exec()
