"""SEGMENTERS MODULE"""


#! IMPORTS

import inspect
import os
import sys
import warnings
from typing import Any, Callable

import cv2
import h5py
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
from numpy.typing import NDArray

from . import assets, utils
from .segmenters import EllipseSegmenter, Segmenter

__all__ = [
    "ndarray2qpixmap",
    "make_shortcut",
    "get_label",
    "FlatButtonWidget",
    "TextSpinBoxWidget",
    "SegmenterWidget",
    "CounterWidget",
    "ResizableImageWidget",
    "FileBrowseBarWidget",
    "SaveBarWidget",
    "QFONT",
    "QSIZE",
    "LabellerWidget",
]

#! GLOBAL VARIABLES


QFONT = qtg.QFont("Arial", 12)
QSIZE = 25

invalid = ["Segmenter", "NDArray", "Signal"]
segmenters_module = sys.modules["heatlab.segmenters"]
SEGMENTERS = inspect.getmembers(segmenters_module, inspect.isclass)
SEGMENTERS = [i for i in SEGMENTERS if i[0] not in invalid]


#! GLOBAL FUNCTIONS


def ndarray2qpixmap(
    ndarray: np.ndarray,
    frmt: qtg.QImage.Format = qtg.QImage.Format_RGBA8888,
):
    """
    return the pixmap corresponding to the NDArray provided.

    Parameters
    ----------
    ndarray: NDArray
        a 3D array with shape and dtype aligned with the given format.
        By default, ndarray is espected to have dtype=uint8 and RGBA color
        distribution.

    frmt: QtGui.QImage.Format
        the image format. By default an RGB format is used.

    Returns
    -------
    qpix: QtGui.QPixmap
        the pixmap corresponding to the array.
    """
    # check the entries
    utils.check_type(ndarray, np.ndarray)
    assert ndarray.ndim >= 3, "ndarray must be a 3D+ NDArray."
    utils.check_type(frmt, qtg.QImage.Format)

    # transform ndarray to pixmap
    shape = ndarray.shape
    qimg = qtg.QImage(
        ndarray,
        shape[1],
        shape[0],
        shape[-1] * shape[1],
        frmt,
    )
    return qtg.QPixmap(qimg)


def make_shortcut(
    shortcut: str | qtc.Qt.Key,
    parent: qtw.QWidget,
    action: Callable,
):
    """
    create a new shortcut.

    Parameters
    ----------
    shortcut: str | Qt.Key
        the shortcut to be linked.

    parent: QWidget
        the widget being the parent of the shortcut.

    action: FunctionType | MethodType
        the action triggered by the shortcut.

    Returns
    -------
    shortcut: QShortcut
        the created shortcut.
    """
    utils.check_type(shortcut, (str, qtc.Qt.Key))
    utils.check_type(parent, qtw.QWidget)
    utils.check_type(action, Callable)
    out = qtw.QShortcut(qtg.QKeySequence(shortcut), parent)
    out.activated.connect(action)
    return out


def get_label(text: str):
    """
    return a QLabel formatted with the given text.

    Parameters
    ----------
    text: str
        The label text.

    Returns
    -------
    label: QtQLabel
        the label object.
    """
    out = qtw.QLabel(text)
    out.setFont(QFONT)
    out.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignVCenter)
    return out


#! CLASSES


class FlatButtonWidget(qtw.QLabel):
    """
    create a custom flat button with the ability to
    discriminate between clicking events.

    Parameters
    ----------
    text: str
        the text to be shown by the button.

    icon: str | QPixmap None
        the name of the icon to be used on this button. The icon is retrieved
        from the "assets" folder of the module.
        If None, no icon is used.

    tooltip: str | None
        the tooltip to be shown by the button. If None, no tooltip is provided.

    click_action: FunctionType | MethodType | None
        the function to be linked to the click action of the button.

    click_shortcut: str
        the action shortcut
    """

    # ****** SIGNALS ****** #

    _clicked: utils.Signal

    # ****** VARIABLES ****** #

    _click_function: Callable
    _click_shortcut: qtw.QShortcut

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        text: str | None = None,
        icon: str | qtg.QPixmap | None = None,
        tooltip: str | None = "",
        fun: Callable | None = None,
        shortcut: str | None = None,
    ):
        super().__init__(text)
        self.setFont(QFONT)
        self.setAlignment(qtc.Qt.AlignCenter | qtc.Qt.AlignVCenter)
        self._clicked = utils.Signal()

        # set the input data
        if icon is not None and not isinstance(icon, qtg.QPixmap):
            if icon is not None and not os.path.exists(icon):
                warnings.warn(f"{icon} does not exists.")
        if tooltip is not None:
            self.setToolTip(tooltip)
        self.setFont(QFONT)

        # connect to the signals
        if fun is not None:
            self.set_click_fun(fun)
        if shortcut is not None:
            self.set_shortcut(shortcut)

        # set icon and tooltip
        self.set_icon(icon)
        self.setToolTip(tooltip)

    # ****** SETTERS ****** #

    def set_icon(self, icon: str | np.ndarray | qtg.QPixmap | None):
        """set the label icon as pixmap."""
        if isinstance(icon, qtg.QPixmap):
            self.setPixmap(icon.scaled(QSIZE, QSIZE))
        elif isinstance(icon, str):
            qpix = assets.as_pixmap(icon)
            self.setPixmap(qpix.scaled(QSIZE, QSIZE))
        elif isinstance(icon, np.ndarray):
            qpix = ndarray2qpixmap(icon)
            self.setPixmap(qpix.scaled(QSIZE, QSIZE))

    def set_click_fun(self, fun: Callable):
        """
        setup the left-click action.

        Parameters
        ----------
        fun: FunctionType | MethodType
            the action to be linked to left-click mouse event.
        """
        utils.check_type(fun, Callable)
        self._click_function = fun
        self._clicked.connect(self._click_function)

    def set_shortcut(self, shortcut: str):
        """set the shortcut linked to the button."""
        self._click_shortcut = make_shortcut(
            shortcut=shortcut,
            parent=self,
            action=self._click_function,
        )

    # ****** PROPERTIES ****** #

    @property
    def icon(self):
        """return the icon installed on the button."""
        return self.pixmap()

    @property
    def click_function(self):
        """return the left click action."""
        return self._click_function

    @property
    def clicked(self):
        """return the left click signal."""
        return self._clicked

    @property
    def shortcut(self):
        """return the shortcut linked to the object."""
        return self._click_shortcut

    @property
    def drop_action(self):
        """drop action"""
        return self._drop_action

    # ****** EVENT HANDLES ****** #

    def mousePressEvent(self, event: qtg.QMouseEvent):
        """handle mouse double clicking event"""
        if event.button() == qtc.Qt.LeftButton:
            self._clicked.emit(self)


class TextSpinBoxWidget(qtw.QWidget):
    """
    create a custom widget where a spinbox is associated to a label and to a
    checkbox.

    Parameters
    ----------
    start_value: int | float
        the starting value of the slider

    min_value: int | float
        the minimum slider value

    max_value: int | float
        the maximum slider value

    step: int | float
        the step size increment/decrement

    decimals: int
        the number of decimals to be used for rendering the data.

    descriptor: str
        the descriptor of the values.
    """

    # ****** SIGNALS ****** #

    _value_changed: utils.Signal
    _state_changed: utils.Signal

    # ****** VARIABLES ****** #

    _label: qtw.QLabel
    _spinbox: qtw.QDoubleSpinBox

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        value: int | float,
        min_value: int | float,
        max_value: int | float,
        step: int | float,
        decimals: int,
        descriptor: str,
    ):
        super().__init__()

        # setup the widget
        self._spinbox = qtw.QDoubleSpinBox()
        self._spinbox.setFont(QFONT)
        self._label = qtw.QLabel()
        self._label.setFont(QFONT)
        self._checkbox = qtw.QCheckBox()
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._label)
        layout.addWidget(self._spinbox)
        self.setLayout(layout)

        # setup the initial values
        self.set_min_value(min_value)
        self.set_max_value(max_value)
        self.set_step(step)
        self.set_decimals(decimals)
        self.set_value(value)
        self.set_descriptor(descriptor)

        # setup the connections
        self._value_changed = utils.Signal()
        self._state_changed = utils.Signal()
        self._spinbox.valueChanged.connect(self._on_value_changed)

    # ****** SETTERS ****** #

    def set_value(self, value: int | float):
        """set the actual value."""
        utils.check_type(value, (int, float))
        val = float(value)
        self._spinbox.setValue(val)

    def set_min_value(self, value: int | float):
        """set the min value."""
        utils.check_type(value, (int, float))
        val = float(value)
        self._spinbox.setMinimum(val)

    def set_max_value(self, value: int | float):
        """set the max value."""
        utils.check_type(value, (int, float))
        val = float(value)
        self._spinbox.setMaximum(val)

    def set_step(self, value: int | float):
        """set the step value."""
        utils.check_type(value, (int, float))
        val = float(value)
        self._spinbox.setSingleStep(val)

    def set_decimals(self, value: int):
        """set the step value."""
        utils.check_type(value, int)
        self._spinbox.setDecimals(value)

    def set_descriptor(self, value: str):
        """set the descriptor."""
        utils.check_type(value, str)
        self._label.setText(value)

    # ****** PROPERTIES ****** #

    @property
    def value(self):
        """return the object value"""
        return self._spinbox.value()

    @property
    def min_value(self):
        """return the object min_value"""
        return self._spinbox.minimum()

    @property
    def max_value(self):
        """return the object max_value"""
        return self._spinbox.maximum()

    @property
    def decimals(self):
        """return the object decimals"""
        return self._spinbox.decimals()

    @property
    def step(self):
        """return the object step"""
        return self._spinbox.singleStep()

    @property
    def descriptor(self):
        """return the object descriptor"""
        return self._label.text()

    @property
    def value_changed(self):
        """return the value_changed signal"""
        return self._value_changed

    # ****** EVENT HANDLERS ****** #

    def _on_value_changed(self, value: float):
        """handle the update of the spinbox value."""
        self.value_changed.emit(value)


class SegmenterWidget(qtw.QWidget):
    """
    Widget allowing the customization of the Segmenter object.

    Parameters
    ----------
    segmenter: Segmenter
        the segmenter linked to the widget.
    """

    # ****** SIGNALS ****** #

    _color_changed: utils.Signal
    _name_changed: utils.Signal
    _linewidth_changed: utils.Signal
    _linewidth_checked: utils.Signal
    _fontsize_changed: utils.Signal
    _fontsize_checked: utils.Signal
    _checked: utils.Signal
    _deleted: utils.Signal
    _move_downward: utils.Signal
    _move_upward: utils.Signal
    _text_checked: utils.Signal

    # ****** VARIABLES ****** #

    _name_button: FlatButtonWidget
    _color_button: FlatButtonWidget
    _linewidth_box: TextSpinBoxWidget
    _fontsize_box: TextSpinBoxWidget
    _delete_button: FlatButtonWidget
    _check_box: qtw.QCheckBox
    _segmenter: Segmenter
    _options_widget: qtw.QWidget
    _upward_button: FlatButtonWidget
    _downward_button: FlatButtonWidget
    _text_checkbox: qtw.QCheckBox

    # ****** CONSTRUCTOR ****** #

    def __init__(self, segmenter: Segmenter):
        super().__init__()

        # signals
        self._color_changed = utils.Signal()
        self._name_changed = utils.Signal()
        self._linewidth_changed = utils.Signal()
        self._linewidth_checked = utils.Signal()
        self._fontsize_changed = utils.Signal()
        self._fontsize_checked = utils.Signal()
        self._checked = utils.Signal()
        self._deleted = utils.Signal()
        self._move_upward = utils.Signal()
        self._move_downward = utils.Signal()
        self._text_checked = utils.Signal()

        # name
        self._name_button = FlatButtonWidget(
            text="SEGMENTER",
            icon=None,
            tooltip="Rename",
            fun=self._name_clicked,
        )
        self._name_button.setFixedHeight(QSIZE)
        align_right = qtc.Qt.AlignRight | qtc.Qt.AlignVCenter
        self._name_button.setAlignment(align_right)

        # color
        self._color_button = FlatButtonWidget(
            text=None,
            icon=None,
            tooltip="Change the color",
            fun=self._color_clicked,
        )
        self._color_button.setFixedSize(QSIZE, QSIZE)

        # check button
        self._check_box = qtw.QCheckBox()
        self._check_box.setCheckable(True)
        self._check_box.setChecked(False)
        self._check_box.setFixedSize(QSIZE, QSIZE)
        self._check_box.setToolTip("Activate/Deactivate the segmenter")
        self._check_box.stateChanged.connect(self._check_clicked)

        # movement box
        self._upward_button = FlatButtonWidget(
            text=None,
            icon=assets.as_pixmap(assets.UPWARD),
            tooltip="Move up",
            fun=self._on_move_upward,
        )
        self._downward_button = FlatButtonWidget(
            text=None,
            icon=assets.as_pixmap(assets.DOWNWARD),
            tooltip="Move down",
            fun=self._on_move_downward,
        )
        move_layout = qtw.QHBoxLayout()
        move_layout.setContentsMargins(0, 0, 0, 0)
        move_layout.setSpacing(0)
        move_layout.addWidget(self._upward_button)
        move_layout.addWidget(self._downward_button)
        move_button = qtw.QWidget()
        move_button.setLayout(move_layout)
        move_button.setFixedSize(2 * QSIZE, QSIZE)

        # main line
        upper_layout = qtw.QHBoxLayout()
        upper_layout.addWidget(move_button)
        upper_layout.addWidget(self._name_button)
        upper_layout.addWidget(self._color_button)
        upper_layout.addWidget(self._check_box)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(5)
        upper_widget = qtw.QWidget()
        upper_widget.setLayout(upper_layout)

        # delete
        self._delete_button = FlatButtonWidget(
            text=None,
            icon=assets.as_pixmap(assets.DELETE),
            tooltip="Remove the Segmenter",
            fun=self._delete_clicked,
        )
        self._delete_button.setFixedSize(QSIZE, QSIZE)

        # linewidth
        self._linewidth_box = TextSpinBoxWidget(1, 1, 20, 1, 0, "Linewidth")
        self._linewidth_box.value_changed.connect(self._on_linewidth_changed)
        self._linewidth_box.setFixedHeight(QSIZE)

        # fontsize
        self._fontsize_box = TextSpinBoxWidget(4, 1, 20, 1, 0, "Fontsize")
        self._fontsize_box.value_changed.connect(self._on_fontsize_changed)
        self._fontsize_box.setFixedHeight(QSIZE)

        # text checkbox
        self._text_checkbox = qtw.QCheckBox()
        self._text_checkbox.setChecked(True)
        self._text_checkbox.setFont(QFONT)
        self._text_checkbox.setToolTip("Enable/disable text view.")
        self._text_checkbox.stateChanged.connect(self._on_text_checked)

        # options layout
        lower_layout = qtw.QHBoxLayout()
        lower_layout.addWidget(self._delete_button)
        lower_layout.addWidget(self._text_checkbox)
        lower_layout.addWidget(self._fontsize_box)
        lower_layout.addWidget(self._linewidth_box)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(5)
        self._options_widget = qtw.QWidget()
        self._options_widget.setLayout(lower_layout)
        self._options_widget.setVisible(False)

        # central layout
        central_layout = qtw.QVBoxLayout()
        central_layout.addWidget(upper_widget)
        central_layout.addWidget(self._options_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(2)
        self.setLayout(central_layout)

        # set the segmenter
        self.set_segmenter(segmenter)

        # set the minimum size
        self.setBaseSize(self.minimumSizeHint())
        self.sizePolicy().setHorizontalPolicy(qtw.QSizePolicy.Minimum)

    # ****** SETTERS ****** #

    def set_segmenter(self, segmenter: Segmenter):
        """
        set a new segmenter for the widget.

        Parameters
        ----------
        segmenter: Segmenter
            a new segmenter.
        """
        utils.check_type(segmenter, Segmenter)
        self._segmenter = segmenter

        # update the widget options
        self.set_color(segmenter.color)
        self.set_name(segmenter.name)
        self.set_linewidth(segmenter.linewidth)
        self.set_fontsize(segmenter.fontsize)

    def set_color(self, rgba: tuple[int, int, int, int]):
        """
        set the required rgba color.

        Parameters
        ----------
        rgb: Iterable | None
            the 4 elements tuple defining a color.
        """

        # check the input
        if rgba is not None:
            msg = "color must be an len=4 iterable of int(s) in the [0-255]"
            msg += " range."
            if not isinstance(rgba, tuple) or not len(rgba) == 4:
                raise TypeError(msg)
            if not all(0 <= i <= 255 for i in rgba):
                raise ValueError(msg)
            if not all(isinstance(i, int) for i in rgba):
                raise ValueError(msg)
        else:
            rgba = tuple(np.random.randint(0, 256) for _ in range(4))

        # generate an ndarray with the given color
        self._segmenter.set_color(rgba)
        self._color_button.setFixedSize(QSIZE, QSIZE)
        self._color_button.set_icon(self.segmenter.icon((QSIZE, QSIZE)))
        self.color_changed.emit(self, rgba)

    def set_name(self, name: str):
        """rename the current label."""
        old = self._name_button.text()
        self._name_button.setText(name)
        self._segmenter.set_name(name)
        self.name_changed.emit(self, old, name)

    def set_linewidth(self, linewidth: int):
        """set the segmenterwidget linewidth reference value."""
        utils.check_type(linewidth, int)
        self._linewidth_box.set_value(linewidth)
        self._segmenter.set_linewidth(linewidth)
        self.linewidth_changed.emit(self, self._linewidth_box.value)

    def set_fontsize(self, fontsize: int):
        """set the segmenterwidget fontsize reference value."""
        utils.check_type(fontsize, int)
        self._fontsize_box.set_value(fontsize)
        self._segmenter.set_fontsize(fontsize)
        self.fontsize_changed.emit(self, self._fontsize_box.value)

    def set_move_upward_enabled(self, enabled: bool):
        """enable/disable the upward move button"""
        utils.check_type(enabled, bool)
        self._upward_button.setEnabled(enabled)

    def set_move_downward_enabled(self, enabled: bool):
        """enable/disable the downward move button"""
        utils.check_type(enabled, bool)
        self._downward_button.setEnabled(enabled)

    # ****** GETTERS ****** #

    @property
    def name_changed(self):
        """return the name change signal"""
        return self._name_changed

    @property
    def color_changed(self):
        """return the color change signal"""
        return self._color_changed

    @property
    def linewidth_changed(self):
        """return the linewidth change signal"""
        return self._linewidth_changed

    @property
    def linewidth_checked(self):
        """return the linewidth check signal"""
        return self._linewidth_checked

    @property
    def fontsize_changed(self):
        """return the fontsize change signal"""
        return self._fontsize_changed

    @property
    def fontsize_checked(self):
        """return the fontsize check signal"""
        return self._fontsize_checked

    @property
    def checked(self):
        """return the checked signal"""
        return self._checked

    @property
    def deleted(self):
        """return the delete signal"""
        return self._deleted

    @property
    def segmenter(self):
        """return the segmenter"""
        return self._segmenter

    @property
    def fontsize_is_checked(self):
        """return the state of the fontsize textspinbox"""
        return self._fontsize_box.is_checked()

    @property
    def linewidth_is_checked(self):
        """return the state of the linewidth textspinbox"""
        return self._linewidth_box.is_checked()

    @property
    def move_downward(self):
        """return the state_changed signal"""
        return self._move_downward

    @property
    def move_upward(self):
        """return the state_changed signal"""
        return self._move_upward

    @property
    def text_checked(self):
        """return the text_checked signal"""
        return self._text_checked

    # ****** EVENT HANDLERS ****** #

    def _color_clicked(self, source: FlatButtonWidget):
        """select and set the desired color."""
        self.change_color()

    def _name_clicked(self, source: FlatButtonWidget):
        """handle the single clicking action."""
        self.change_name()

    def _check_clicked(self, source: FlatButtonWidget):
        """handle the double clicking action on the segmenter widget"""
        self.checked.emit(self)

    def _delete_clicked(self, source: FlatButtonWidget):
        """handle the delete button clicking."""
        self.deleted.emit(self)

    def _on_linewidth_changed(self, value: int | float):
        """handle the linewidth change."""
        self.segmenter.set_linewidth(int(value))
        self.linewidth_changed.emit(self, int(value))

    def _on_linewidth_checked(self, value: bool):
        """handle the state of the linewidth SpinCheck"""
        self.linewidth_checked.emit(self, value)

    def _on_fontsize_changed(self, value: int | float):
        """handle the fontsize change."""
        self.segmenter.set_fontsize(int(value))
        self.fontsize_changed.emit(self, int(value))

    def _on_fontsize_checked(self, value: bool):
        """handle the state of the fontsize SpinCheck"""
        self.fontsize_checked.emit(self, value)

    def _on_move_upward(self, source: FlatButtonWidget):
        """handle the upward button action."""
        self.move_upward.emit(self)

    def _on_move_downward(self, source: FlatButtonWidget):
        """handle the downward button action."""
        self.move_downward.emit(self)

    def _on_text_checked(self, value: bool):
        """enable/disable the fontsize TextSpinbox"""
        self._fontsize_box.setEnabled(self._text_checkbox.isChecked())
        self.text_checked.emit(self)

    def enterEvent(self, event: qtc.QEvent):
        """make the delete button visible."""
        self._options_widget.setVisible(True)
        return super().enterEvent(event)

    def leaveEvent(self, event: qtc.QEvent):
        """hide the delete button."""
        self._options_widget.setVisible(False)
        return super().leaveEvent(event)

    # ****** METHODS ****** #

    def minimumSizeHint(self):
        """return the minimum size hint for this widget."""
        width = self._options_widget.minimumSizeHint().width()
        height = super().minimumSizeHint().height()
        return qtc.QSize(width, height)

    def is_checked(self):
        """return whether the Segmenter is active or not."""
        return self._check_box.isChecked()

    def is_text_enabled(self):
        """return whether the text_checkbox is checked or not."""
        return self._text_checkbox.isChecked()

    def change_color(self):
        """method allowing the change of the Segmenter color."""
        color = qtw.QColorDialog.getColor(
            initial=qtg.QColor(*self.segmenter.color),
        )
        if color.isValid():
            tup = color.red(), color.green(), color.blue(), color.alpha()
            self.set_color(tup)

    def change_name(self):
        """method allowing the change of the Segmenter name."""
        text, done = qtw.QInputDialog.getText(
            self,
            "rename",
            "Write here the new label name",
        )
        if done:
            self.set_name(text)

    def click(self):
        """force the check/uncheck of the segmenter"""
        state = False if self._check_box.isChecked() else True
        self._check_box.setChecked(state)
        self.checked.emit(self)


class SegmenterPaneWidget(qtw.QWidget):
    """
    SegmenterWidget(s) grouper.

    Parameters
    ----------
    segmenters: Iterable[Segmenter] | Segmenter | None
        a list of segmenters to be rendered.
    """

    # ****** SIGNALS ****** #

    _added: utils.Signal
    _removed: utils.Signal
    _text_checked: utils.Signal

    # ****** VARIABLES ****** #

    _segmenters = []
    _add_button: qtw.QPushButton
    _container: qtw.QWidget
    _shortcuts = []
    _scroll: qtw.QScrollArea

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        segmenters_list: list[Segmenter] | Segmenter | None = None,
    ):
        super().__init__()
        self._added = utils.Signal()
        self._removed = utils.Signal()
        self._text_checked = utils.Signal()
        self._shortcuts = []

        # setup the add button
        self._add_button = qtw.QPushButton("Add New")
        self._add_button.setToolTip("Add a new Segmenter.")
        self._add_button.clicked.connect(self._add_clicked)
        self._add_button.setFont(QFONT)
        self._add_button.setFixedHeight(QSIZE)
        min_width = self._add_button.minimumSizeHint().width()
        self._add_button.setFixedWidth(min_width)

        # set the top widget
        descriptor = qtw.QLabel("SEGMENTER LIST")
        descriptor.setFont(QFONT)
        descriptor.setFixedHeight(QSIZE)
        descriptor.setAlignment(qtc.Qt.AlignLeft | qtc.Qt.AlignVCenter)
        top_layout = qtw.QHBoxLayout()
        top_layout.addWidget(descriptor)
        top_layout.addWidget(self._add_button)
        top_layout.setSpacing(5)
        top_widget = qtw.QWidget()
        top_widget.setLayout(top_layout)
        top_widget.setFixedHeight(top_widget.minimumSizeHint().height())

        # set the segmenters widget
        sgm_layout = qtw.QVBoxLayout()
        sgm_layout.setContentsMargins(0, 0, 0, 0)
        sgm_layout.setSpacing(5)
        self._container = qtw.QWidget()
        self._container.setLayout(sgm_layout)

        # setup a scroll area for the container
        column = qtw.QWidget()
        column.setFixedWidth(20)
        inner_layout = qtw.QHBoxLayout()
        inner_layout.setSpacing(0)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(self._container)
        inner_layout.addWidget(column)
        inner_widget = qtw.QWidget()
        inner_widget.setLayout(inner_layout)
        self._scroll = qtw.QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(inner_widget)
        self._scroll.verticalScrollBar().setFixedWidth(20)

        # add the segmenters
        if segmenters_list is not None:
            sgm = (
                segmenters_list
                if isinstance(segmenters_list, list)
                else [segmenters_list]
            )
            for segmenter in sgm:
                utils.check_type(segmenter, Segmenter)
                self.add_segmenter(segmenter)
        self._update_shortcuts()
        self._update_move_buttons()

        # setup the widget layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(top_widget)
        layout.addWidget(self._scroll)
        self.setLayout(layout)

    # ****** GETTERS ****** #

    @property
    def shortcuts(self):
        """return the list of shortcuts."""
        return self._shortcuts

    @property
    def added(self):
        """return the added signal"""
        return self._added

    @property
    def removed(self):
        """return the removed signal"""
        return self._removed

    @property
    def segmenters(self):
        """return the list of available segmenter"""
        return self._segmenters

    @property
    def active_id(self):
        """return the index of the actual active SegmenterWidget or None."""
        active = np.where([i.is_checked() for i in self.segmenters])[0]
        return active[0] if len(active) > 0 else None

    @property
    def active_widget(self):
        """return the actual active SegmenterWidget or None."""
        idx = self.active_id
        return self.segmenters[idx] if idx is not None else None

    @property
    def active_segmenter(self):
        """return the actual active Segmenter or None."""
        segmenter = self.active_widget
        if segmenter is not None:
            segmenter = segmenter.segmenter
        return segmenter

    @property
    def text_checked(self):
        """return the text_checked signal"""
        return self._text_checked

    # ****** EVENT HANDLERS ****** #

    def _update_state(self, source: SegmenterWidget):
        """handle the clicking of any SegmenterWidget within the group."""
        if source.is_checked():
            for i in self.segmenters:
                if i != source and i.is_checked():
                    i.click()
                    # i._check_box.setChecked(False)

    def _delete_segmenter(self, source: FlatButtonWidget):
        """handle the removal of a SegmenterWidget from the group."""

        # remove the segmenter from the list
        index = [i for i, v in enumerate(self.segmenters) if v == source]
        if len(index) > 0:
            self.delete_segmenter(index[0])

        # update the layout
        self._update_layout()

    def _add_clicked(self, source: FlatButtonWidget):
        """handle the removal of a SegmenterWidget from the group."""
        self.new_segmenter()

    def _shift_up(self, source: SegmenterWidget):
        """handle the shift up of the segmenter"""
        index = None
        for i, v in enumerate(self.segmenters):
            if v.segmenter.name == source.segmenter.name:
                index = i
                break
        if index is not None and index > 0:
            pre = self.segmenters[: (index - 1)]
            sgm = [self.segmenters[index]]
            pst = [self.segmenters[index - 1]] + self.segmenters[(index + 1) :]
            self._segmenters = pre + sgm + pst
            self._update_move_buttons()
        self._update_layout()

    def _shift_down(self, source: SegmenterWidget):
        """handle the shift down of the segmenter"""
        index = None
        for i, v in enumerate(self.segmenters):
            if v.segmenter.name == source.segmenter.name:
                index = i
                break
        if index is not None and index < len(self.segmenters) - 1:
            pre = self.segmenters[:index] + [self.segmenters[index + 1]]
            sgm = [self.segmenters[index]]
            pst = self.segmenters[index + 2 :]
            self._segmenters = pre + sgm + pst
            self._update_move_buttons()
        self._update_layout()

    def _on_text_checked(self, source: SegmenterWidget):
        """handle the check/uncheck of the text_checkbox of one widget"""
        self.text_checked.emit(source)

    # ****** METHODS ****** #

    def _update_layout(self):
        """update the layout according to the actual segmenters listed."""

        # update the layout
        layout = self._container.layout()
        while layout.count() > 0:
            item = layout.itemAt(0)
            layout.removeItem(item)
        for widget in self.segmenters:
            layout.addWidget(widget)
        layout.addStretch()
        self._update_shortcuts()
        self._update_move_buttons()

        if len(self.segmenters) > 0:
            width = max([i.minimumSizeHint().width() for i in self.segmenters])
        else:
            width = self._scroll.minimumSizeHint().width()
        self._scroll.setMinimumWidth(width + 20)

    def new_segmenter(self):
        """add a novel segmenter"""
        # let the user select the new segmenter type
        items = [i[0] for i in SEGMENTERS]
        item, done = qtw.QInputDialog.getItem(
            self,
            "Segmenter type",
            "Set the new segmenter type",
            items,
            0,
            False,
        )

        # add the segmenter
        if done and isinstance(item, str):
            idx = np.where([i == item for i in items])[0][0]
            cls = SEGMENTERS[idx][1]
            self.add_segmenter(
                segmenter=cls(name=f"SEGMENTER{len(self.segmenters) + 1}"),
                index=0,
            )
            self.segmenters[0].click()

    def add_segmenter(
        self,
        segmenter: Segmenter,
        index: int | None = None,
    ):
        """
        add a SegmenterWidget to the group.

        Parameters
        ----------
        segmenter: Segmenter
            a Segmenter object to be appended to the group.

        index: int | None
            the position of the list at which include the Segmenter.
        """

        # check the entries
        utils.check_type(segmenter, Segmenter)
        isin = any(segmenter.name == i.segmenter.name for i in self.segmenters)
        if not isin:
            # convert the segmenter to SegmenterWidget and
            # connect with _update_state and _delete_segmenter actions
            widget = SegmenterWidget(segmenter)
            widget.checked.connect(self._update_state)
            widget.deleted.connect(self._delete_segmenter)
            widget.move_upward.connect(self._shift_up)
            widget.move_downward.connect(self._shift_down)
            widget.text_checked.connect(self._on_text_checked)

            # append the segmenter
            if index is None:
                self._segmenters += [widget]
            elif isinstance(index, int):
                self._segmenters.insert(index, widget)
            else:
                raise TypeError(f"{index} must be an {int} instance.")

            # update
            self._update_layout()
            self.added.emit(widget)

    def delete_segmenter(self, index: int):
        """
        delete a SegmenterWidget from the group.

        Parameters
        ----------
        index: int
            the position of the widget to be removed from the list.
        """
        if isinstance(index, int):
            wdg = self.segmenters.pop(index)
            wdg.setVisible(False)
            self._update_layout()
            self.removed.emit(wdg)
        else:
            raise TypeError(f"{index} must be an {int} instance.")

    def _update_shortcuts(self):
        """update the shortcuts linked to the object."""
        for shortcut in self._shortcuts:
            shortcut.deleteLater()
        self._shortcuts = []
        for i, segmenter in enumerate(self.segmenters):
            if i == 9:
                msg = "No more than 9 segmenters can be linked to numerical "
                msg += "shortcuts."
                qtw.QMessageBox.warning(self, "WARNING", msg)
            elif i < 9:
                key = qtc.Qt.Key(qtc.Qt.Key_1 + i)
                short = make_shortcut(key, self, segmenter.click)
                self._shortcuts += [short]
        add_short = make_shortcut(qtc.Qt.Key_Plus, self, self.new_segmenter)
        self._shortcuts += [add_short]

    def _update_move_buttons(self):
        """
        update the move button are enabled/disabled according to the position
        of the segmenter in the list.
        """
        n = len(self.segmenters)
        for i, segmenter in enumerate(self.segmenters):
            segmenter.set_move_downward_enabled(i < n - 1)
            segmenter.set_move_upward_enabled(i > 0)


class CounterWidget(qtw.QWidget):
    """
    Image counter widget.

    Parameters
    ----------
    max_counter: int
        the maximum number of elements to be counted.

    start_value: int | None
        the starting counter value. if None, it is initialized to zero.
    """

    # ****** SIGNALS ****** #

    _move_forward: utils.Signal
    _move_backward: utils.Signal

    # ****** VARIABLES ****** #

    _label: qtw.QLabel
    _forward_button: qtw.QPushButton
    _backward_button: qtw.QPushButton
    _counter_label: qtw.QLabel
    _max_counter_label: qtw.QLabel

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        text: str,
        max_counter: int,
        start_value: int | None = None,
    ):
        super().__init__()
        self._move_forward = utils.Signal()
        self._move_backward = utils.Signal()

        # setup the max counter
        utils.check_type(max_counter, int)
        self._max_counter_label = get_label(str(max_counter))

        # setup the counter
        self._counter_label = get_label("")
        if start_value is None:
            self.set_counter(1)
        else:
            utils.check_type(start_value, int)
            self.set_counter(start_value)

        # setup the text
        utils.check_type(text, str)
        self._label = get_label(text)

        # setup the forward button
        size = self._counter_label.minimumSizeHint().height()
        self._forward_button = qtw.QPushButton()
        self._forward_button.setShortcut(qtg.QKeySequence(qtc.Qt.Key_Right))
        self._forward_button.clicked.connect(self._forward_pressed)
        forward_pixmap = assets.as_pixmap(assets.FORWARD).scaled(size, size)
        forward_icon = qtg.QIcon(forward_pixmap)
        self._forward_button.setIcon(forward_icon)
        self._forward_button.setFixedSize(size, size)
        self._forward_button.setToolTip("Next Frame")
        self._forward_button.setFont(QFONT)

        # setup the backward button
        self._backward_button = qtw.QPushButton()
        self._backward_button.setShortcut(qtg.QKeySequence(qtc.Qt.Key_Left))
        self._backward_button.clicked.connect(self._backward_pressed)
        backward_pixmap = assets.as_pixmap(assets.BACKWARD).scaled(size, size)
        backward_icon = qtg.QIcon(backward_pixmap)
        self._backward_button.setIcon(backward_icon)
        self._backward_button.setFixedSize(size, size)
        self._backward_button.setToolTip("Previous Frame")
        self._backward_button.setFont(QFONT)
        self._backward_button.setEnabled(False)

        # setup the whole widget layout
        layout = qtw.QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self._label)
        layout.addStretch()
        layout.addWidget(self._backward_button)
        layout.addWidget(self._counter_label)
        layout.addWidget(get_label(" / "))
        layout.addWidget(self._max_counter_label)
        layout.addWidget(self._forward_button)
        layout.addStretch()
        self.setLayout(layout)

    # ****** SETTERS ****** #

    def set_counter(self, cnt: int):
        """set the counter to the required value."""
        utils.check_type(cnt, int)
        digits = len(self._max_counter_label.text())
        frm = "{:0" + str(digits) + "d}"
        self._counter_label.setText(frm.format(cnt))

    def set_max_counter(self, cnt: int):
        """set the max counter to the required value."""
        utils.check_type(cnt, int)
        self._max_counter_label.setText(str(cnt))

    # ****** GETTERS ****** #

    @property
    def move_forward(self):
        """return the move forward signal"""
        return self._move_forward

    @property
    def move_backward(self):
        """return the move backward signal"""
        return self._move_backward

    @property
    def counter(self):
        """return the actual counter value."""
        return int(self.counter_label.text())

    @property
    def max_counter(self):
        """return the maximum value achievable."""
        return int(self.max_counter_label.text())

    @property
    def max_counter_label(self):
        """return the max counter QLabel."""
        return self._max_counter_label

    @property
    def counter_label(self):
        """return the counter QLabel."""
        return self._counter_label

    # ****** EVENT HANDLERS ****** #

    def _forward_pressed(self):
        """handle the clicking of the forward button."""
        if self.counter < self.max_counter:
            self.set_counter(self.counter + 1)
            self.move_forward.emit()
        self._forward_button.setEnabled(self.counter < self.max_counter)
        self._backward_button.setEnabled(self.counter > 1)

    def _backward_pressed(self):
        """handle the clicking of the backward button."""
        if self.counter > 0:
            self.set_counter(self.counter - 1)
            self.move_backward.emit()
        self._forward_button.setEnabled(self.counter < self.max_counter)
        self._backward_button.setEnabled(self.counter > 1)


class ResizableImageWidget(qtw.QWidget):
    """create a widget accepting a 3D numpy array to be rendered as image."""

    # ****** SIGNALS ****** #

    _image_changed: utils.Signal
    _mouse_pressed: utils.Signal
    _mouse_released: utils.Signal
    _mouse_doubleclick: utils.Signal
    _mouse_moved: utils.Signal
    _mouse_enter: utils.Signal
    _mouse_leave: utils.Signal
    _mouse_wheeling: utils.Signal

    # ****** VARIABLES ****** #

    _ndarray: NDArray
    _scale: int | float
    _image_coords: tuple[int, int] | None
    _scroll: qtw.QScrollArea
    _label: qtw.QLabel

    # ****** CONSTRUCTOR ****** #

    def __init__(self):
        super().__init__()

        # setup the widget
        self._label = qtw.QLabel()
        self._scroll = qtw.QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        policy = self._scroll.sizePolicy()
        policy.setHorizontalPolicy(qtw.QSizePolicy.Ignored)
        policy.setVerticalPolicy(qtw.QSizePolicy.Ignored)
        self._scroll.setSizePolicy(policy)
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(True)
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._scroll)
        self.setLayout(layout)

        # setup the event tracking
        self._label.setMouseTracking(True)
        self._label.installEventFilter(self)
        self._image_changed = utils.Signal()
        self._mouse_pressed = utils.Signal()
        self._mouse_released = utils.Signal()
        self._mouse_doubleclick = utils.Signal()
        self._mouse_moved = utils.Signal()
        self._mouse_enter = utils.Signal()
        self._mouse_leave = utils.Signal()
        self._mouse_wheeling = utils.Signal()

        # adjust the size
        self._scroll.resize(self._scroll.viewport().size())

    # ****** SETTERS ****** #

    def set_coords(self, pos: qtc.QPoint | None):
        """coords
        set the coordinates of the mouse pointer and update
        the data coords labels.

        Parameters
        ----------
        pos: QPoint | None
            the mouse position from which the coordinates are extracted.
        """
        if pos is None:
            self._image_coords = None
        else:
            utils.check_type(pos, qtc.QPoint)
            mouse_x, mouse_y = pos.x(), pos.y()
            image_shape = self._ndarray.shape[:2][::-1]
            image_y = int(image_shape[1] * self._scale)
            space_y = self._scroll.height()
            y_off = (space_y - image_y) // 2
            if y_off > 0:
                self._image_coords = (mouse_x, mouse_y - y_off)
            else:
                self._image_coords = (mouse_x, mouse_y)

    def set_ndarray(self, ndarray: np.ndarray):
        """
        update the image with the provided ndarray.

        Parameters
        ----------
        ndarray: NDArray
            a 3D ndarray of the image to be rendered. The array must have
            dtype uint8 and should be designed according to the RGBA format.
        """
        utils.check_type(ndarray, np.ndarray)
        assert ndarray.ndim == 3, "ndarray must be a 3D NDArray."
        assert ndarray.dtype == np.uint8, "ndarray must have dtype uint8."
        self._ndarray = ndarray

        # update the view as appropriate
        self.update_view()

    def set_scale(self, scale: float | int):
        """
        set the scaling factor between the raw image and the visualized one.

        Parameters
        ----------
        scale: int | float
            the scaling factor.
        """
        utils.check_type(scale, (int, float))
        self._scale = scale

    # ****** GETTERS ****** #

    @property
    def mouse_enter(self):
        """return the mouse_enter signal"""
        return self._mouse_enter

    @property
    def mouse_leave(self):
        """return the mouse_leave signal"""
        return self._mouse_leave

    @property
    def mouse_pressed(self):
        """return the mouse_press signal"""
        return self._mouse_pressed

    @property
    def mouse_doubleclick(self):
        """return the mouse_doubleclick signal"""
        return self._mouse_doubleclick

    @property
    def mouse_moved(self):
        """return the mouse_moved signal"""
        return self._mouse_moved

    @property
    def mouse_released(self):
        """return the mouse_released signal"""
        return self._mouse_released

    @property
    def mouse_wheeling(self):
        """return the mouse_wheeling signal"""
        return self._mouse_wheeling

    @property
    def image_changed(self):
        """return the image_changed signal"""
        return self._image_changed

    @property
    def ndarray(self):
        """return the ndarray defining the image."""
        return self._ndarray

    @property
    def scale(self):
        """
        return the scaling factor between the raw image and the visualized one.
        """
        return self._scale

    @property
    def image_coords(self):
        """return the (x, y) coordinates of the mouse in pixels."""
        return self._image_coords

    @property
    def data_coords(self):
        """return the (x, y) coordinates of the mouse in data units."""
        if self.image_coords is None or self.scale is None:
            return None
        x, y = tuple(int(i // self.scale) for i in self.image_coords)
        return x, y

    # ****** EVENT HANDLERS ****** #

    def eventFilter(self, obj: qtc.QObject, evt: qtc.QEvent):
        """event filter handler"""
        is_under = isinstance(evt, qtg.QMouseEvent) and self.is_under_mouse(evt)

        # mouse wheeling
        if evt.type() == qtc.QEvent.Wheel:
            self._on_mouse_wheeling(evt)

        # mouse double-click
        if evt.type() == qtc.QEvent.MouseButtonDblClick and is_under:
            self._on_mouse_doubleclick(evt)
            return True

        # mouse press
        if (
            evt.type() == qtg.QMouseEvent.MouseButtonPress
            and evt.buttons() == qtc.Qt.LeftButton
            and is_under
        ):
            self._on_mouse_press(evt)
            return True

        # mouse release
        if evt.type() == qtg.QMouseEvent.MouseButtonRelease and is_under:
            self._on_mouse_release(evt)
            return True

        # mouse move
        if evt.type() == qtg.QMouseEvent.MouseMove:
            if is_under:
                self._on_mouse_move(evt)
            else:
                self._on_mouse_leave(evt)
            return True

        # enter
        if evt.type() == qtc.QEvent.Enter and is_under:
            self._on_mouse_enter(evt)
            return True

        # leave
        if evt.type() == qtc.QEvent.Leave:
            self._on_mouse_leave(evt)
            return True

        # return
        return super().eventFilter(obj, evt)

    def _on_mouse_press(self, event: qtg.QMouseEvent):
        """mouse press event."""
        self.set_coords(event.pos())
        self.mouse_pressed.emit(self.data_coords)

    def _on_mouse_doubleclick(self, event: qtg.QMouseEvent):
        """mouse double-click event."""
        self.set_coords(None)
        self.mouse_doubleclick.emit(self.data_coords)

    def _on_mouse_move(self, event: qtg.QMouseEvent):
        """mouse move event."""
        self.set_coords(event.pos())
        self.mouse_moved.emit(self.data_coords)

    def _on_mouse_release(self, event: qtg.QMouseEvent):
        """mouse release event."""
        self.set_coords(event.pos())
        self.mouse_released.emit(self.data_coords)

    def _on_mouse_enter(self, event: qtg.QMouseEvent):
        """image entering event"""
        self.set_coords(event.pos())
        self.mouse_enter.emit(self.data_coords)

    def _on_mouse_leave(self, event: qtg.QMouseEvent):
        """image leaving event"""
        self.set_coords(None)
        self.mouse_leave.emit(self.data_coords)

    def _on_mouse_wheeling(self, event: qtc.QEvent.Wheel):
        """mouse wheeling action"""
        self.set_coords(event.pos())
        self.mouse_wheeling.emit(event.angleDelta().y() / 15)

    def resizeEvent(self, event: qtg.QResizeEvent):
        """image resizing event"""
        super().resizeEvent(event)
        self._scroll.viewport().resize(self._scroll.size())
        self.update_view()

    # ****** METHODS ****** #

    def is_under_mouse(self, event: qtg.QMouseEvent):
        """return if the label is under the mouse."""
        if self.ndarray is None or self.scale is None:
            return False
        mouse = event.pos()
        mouse_x, mouse_y = mouse.x(), mouse.y()
        image_shape = self._ndarray.shape[:2][::-1]
        image_x, image_y = tuple(int(i * self._scale) for i in image_shape)
        space_y = self._label.height()
        y_off = (space_y - image_y) // 2
        x_on = mouse_x <= image_x
        if y_off > 0:
            y_on = mouse_y > y_off and mouse_y <= image_y + y_off
        else:
            y_on = mouse_y <= image_y
        return x_on and y_on

    def update_scale(self):
        """
        update the scaling factor between the ndarray and the
        available widget space.
        """
        hint = self._scroll.size()
        hint = (hint.width(), hint.height())
        raw = self._ndarray.shape[:2][::-1]
        self.set_scale(min(i / v for i, v in zip(hint, raw)))

    def update_view(self):
        """update the image with the provided ndarray."""
        self.update_scale()
        dsize = tuple(int(i * self.scale) for i in self._ndarray.shape[:2][::-1])
        interp = cv2.INTER_LINEAR
        img = cv2.resize(src=self.ndarray, dsize=dsize, interpolation=interp)
        self._label.setPixmap(ndarray2qpixmap(img))
        self.image_changed.emit(self)


class FileBrowseBarWidget(qtw.QWidget):
    """
    widget allowing to set the path to a file on the system.

    Parameters
    ----------
    formats: Iterable[str]
        the list of accepted formats.
    """

    # ****** SIGNALS ****** #

    _text_changed: utils.Signal

    # ****** VARIABLES ****** #

    _textfield: qtw.QTextEdit
    _browse_button: qtw.QPushButton
    _formats = []
    _last_path = __file__.rsplit(os.path.sep, 1)[0]

    # ****** CONSTRUCTOR ****** #

    def __init__(self, formats: list | None):
        super().__init__()

        # setup the widget
        self._browse_button = qtw.QPushButton("BROWSE")
        self._browse_button.setFont(QFONT)
        size = self._browse_button.sizeHint()
        width = size.width()
        height = size.height()
        self._browse_button.setFixedSize(width + 5, height + 5)
        self._browse_button.clicked.connect(self._on_browse_press)
        self._textfield = qtw.QTextEdit("")
        self._textfield.setFont(QFONT)
        self._textfield.setEnabled(False)
        self._textfield.textChanged.connect(self._on_text_changed)
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._textfield)
        layout.addWidget(self._browse_button)
        self.setLayout(layout)

        # setup the event tracking
        self._text_changed = utils.Signal()
        self.setFixedHeight(height + 5)

        # setup the accepted formats
        if formats is not None:
            self.set_formats(formats)

    # ****** SETTERS ****** #

    def set_formats(self, formats: list):
        """
        Set the accepted file formats from the file browser

        Parameters
        ----------
        formats: Iterable[str]
            an iterable containing the accepted file formats.
        """
        self._formats = []

        # check the entries
        utils.check_type(formats, list)
        for ext in formats:
            utils.check_type(ext, str)
            self._formats += [ext]

    # ****** GETTERS ****** #

    @property
    def text_changed(self):
        """return the file_changed signal"""
        return self._text_changed

    @property
    def text(self):
        """return the mouse_leave signal"""
        return self._textfield.toPlainText()

    @property
    def formats(self):
        """return the list of accepted_formats."""
        return self._formats

    # ****** EVENT HANDLERS ****** #

    def _on_browse_press(self):
        """browse button press event."""
        file = qtw.QFileDialog.getOpenFileName(
            self,
            "Select File",
            self._last_path,
            "formats (" + " ".join([f"*.{i}" for i in self.formats]) + ")",
        )
        if file is not None:
            file = file[0].replace("/", os.path.sep)
            self._last_path = file.rsplit(os.path.sep, 1)[0]
            self._textfield.setText(file)

    def _on_text_changed(self):
        """text change event."""
        if not os.path.exists(self.text):
            qtw.QMessageBox.warning(
                self,
                "File not found",
                f"{self.text} not found.",
            )
        else:
            self.text_changed.emit(self.text)


class SaveBarWidget(qtw.QWidget):
    """
    widget dedicated to highlight the progress of an action triggered by
    pressing one button.

    Parameters
    ----------
    minimum: int | float
        the minimum value of the progress bar.

    maximum: int | float
        the maximum value of the progress bar.

    step: int | float
        the step increment of the progress bar.
    """

    # ****** SIGNALS ****** #

    _started: utils.Signal
    _completed: utils.Signal

    # ****** VARIABLES ****** #

    _progress: qtw.QProgressBar
    _button: qtw.QPushButton
    _step: int | float

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        minimum: int | float,
        maximum: int | float,
        step: int | float,
    ):
        super().__init__()

        # setup the widget
        self._button = qtw.QPushButton("SAVE")
        self._button.setFont(QFONT)
        size = self._button.sizeHint()
        width = size.width()
        height = size.height()
        self._button.setFixedSize(width + 5, height + 5)
        self._button.clicked.connect(self._on_button_press)
        self._progress = qtw.QProgressBar()
        self._progress.setFont(QFONT)
        self._progress.setTextVisible(False)
        self._progress.valueChanged.connect(self._on_value_changed)
        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._progress)
        layout.addWidget(self._button)
        self.setLayout(layout)
        self.setFixedHeight(height + 5)

        # setup the event tracking
        self._started = utils.Signal()
        self._completed = utils.Signal()

        # setters
        self.set_minimum(minimum)
        self.set_maximum(maximum)
        self.set_step(step)

    # ****** SETTERS ****** #

    def set_minimum(self, value: int | float):
        """
        set the minimum value of the progress bar.

        Parameters
        ----------
        minimum: int | float
            the minimum value to be set.
        """
        utils.check_type(value, (int, float))
        self._progress.setMinimum(value)
        self.reset()

    def set_maximum(self, value: int | float):
        """
        set the maximum value of the progress bar.

        Parameters
        ----------
        maximum: int | float
            the maximum value to be set.
        """
        utils.check_type(value, (int, float))
        self._progress.setMaximum(value)
        self.reset()

    def set_step(self, value: int | float):
        """
        set the step value of the progress bar.

        Parameters
        ----------
        maximum: int | float
            the maximum value to be set.
        """
        utils.check_type(value, (int, float))
        assert value > 0, "step must be > 0"
        self._step = value
        self.reset()

    # ****** GETTERS ****** #

    @property
    def started(self):
        """return the start signal"""
        return self._started

    @property
    def completed(self):
        """return the completed signal"""
        return self._completed

    @property
    def minimum(self):
        """return the minimum value."""
        return self._progress.minimum()

    @property
    def maximum(self):
        """return the maximum value."""
        return self._progress.maximum()

    @property
    def step(self):
        """return the step value."""
        return self._step

    @property
    def value(self):
        """return the value of the progress bar"""
        return self._progress.value()

    # ****** EVENT HANDLERS ****** #

    def _on_button_press(self):
        """browse button press event."""
        self._button.setEnabled(False)
        self._progress.setTextVisible(True)
        self.started.emit()

    def _on_value_changed(self):
        """handle the change of the progress bar status"""
        if self.value == self.maximum:
            self.reset()
            self.completed.emit()

    # ****** METHODS ****** #

    def reset(self):
        """reset the values of the progress bar"""
        self._progress.setValue(self.minimum)
        self._button.setEnabled(True)
        self._progress.setTextVisible(False)

    def update(self):
        """
        update the progress
        """
        self._progress.setValue(min(self.value + self.step, self.maximum))


class LabellerWidget(qtw.QWidget):
    """
    GUI allowing to interact with the provided labeller and images

    Parameters
    ----------
    segmenters: Iterable[Segmenter] | Segmenter | None, optional
        pregenerated segmenters to be included in the Labeller.

    **formats: Keyworded arguments
        any number of keyworded arguments. Each should be the name of an
        extension file format accepted by the labeller. Each key must have
        an associated Method or Function that allows to extract the frames
        to be labelled from the selected file.
    """

    # ****** VARIABLES ****** #

    _image_widget: ResizableImageWidget  # the image rendering widget
    _segmenter_pane: SegmenterPaneWidget  # the Widget containing the segmenters
    _counter_widget: CounterWidget  # the Widget containing the position counters
    _options_pane: qtw.QWidget  # the widget containing all the options
    _pressed: bool = False  # a state variable used to deal with mouse pressing
    _input_bar: FileBrowseBarWidget  # file input bar
    _save_bar: SaveBarWidget  # save file button.
    _data_coords: qtw.QLabel  # coordinates label.
    _frames: NDArray  # the images to be labelled
    _segmenters: dict[str, list[Segmenter]]  # the segmenters to be used.
    _formats: dict[str, Callable]  # the accepted file formats.
    _last_path = __file__.rsplit(os.path.sep, 1)[0]
    _estimator_checkbox: qtw.QCheckBox
    _timer: qtc.QTimer

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        segmenters: Segmenter | list[Segmenter] | None = None,
        **formats,
    ):
        super().__init__()

        # setup the title and app icon
        self.setWindowIcon(qtg.QIcon(assets.as_pixmap(assets.ICON)))
        self.setWindowTitle("HeatmapLabeller")

        # setup the image widget
        self._image_widget = ResizableImageWidget()
        self._image_widget.mouse_enter.connect(self._on_enter)
        self._image_widget.mouse_leave.connect(self._on_leave)
        self._image_widget.mouse_moved.connect(self._on_move)
        self._image_widget.mouse_pressed.connect(self._on_press)
        self._image_widget.mouse_released.connect(self._on_release)
        self._image_widget.mouse_doubleclick.connect(self._on_doubleclick)
        self._image_widget.mouse_wheeling.connect(self._on_wheeling)

        # setup the input bar
        if len(formats) == 0:
            formats = {i: utils.read_images for i in utils.IMAGE_FORMATS}
        self.set_formats(formats)
        self._input_bar = FileBrowseBarWidget(list(self.formats.keys()))
        self._input_bar.text_changed.connect(self._on_input_text_changed)
        input_layout = qtw.QHBoxLayout()
        input_layout.addWidget(get_label("SOURCE:"))
        input_layout.addWidget(self._input_bar)
        input_widget = qtw.QWidget()
        input_widget.setLayout(input_layout)
        min_height = input_widget.minimumSizeHint().height()
        input_widget.setFixedHeight(min_height)

        # set the coordinates bar
        self._data_coords = get_label("")
        self._estimator_checkbox = qtw.QCheckBox("ESTIMATOR ENABLED")
        self._estimator_checkbox.setFont(QFONT)
        self._estimator_checkbox.setChecked(True)
        bar_layout = qtw.QHBoxLayout()
        bar_layout.addWidget(self._estimator_checkbox)
        bar_layout.addStretch()
        bar_layout.addWidget(get_label("(x, y, z): "))
        bar_layout.addWidget(self._data_coords)
        bar_widget = qtw.QWidget()
        bar_widget.setLayout(bar_layout)
        bar_widget.setFixedHeight(min_height)

        # setup the save bar
        self._save_bar = SaveBarWidget(0, 6, 1)
        self._save_bar.started.connect(self._on_save_pressed)
        self._save_bar.completed.connect(self._on_save_completed)

        # setup the right side widget
        right_layout = qtw.QVBoxLayout()
        right_layout.addWidget(input_widget)
        right_layout.addWidget(bar_widget)
        right_layout.addWidget(self._image_widget)
        right_layout.addWidget(self._save_bar)
        right_widget = qtw.QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMinimumSize(300, 300)

        # setup the Segmenters Panel
        self._segmenter_pane = SegmenterPaneWidget()
        self._segmenter_pane.added.connect(self._on_added_segmenter)
        self._segmenter_pane.removed.connect(self._on_removed_segmenter)

        # setup the counter box
        self._counter_widget = CounterWidget("FRAME", 0, 0)
        self._counter_widget.move_backward.connect(self._on_backward_pressed)
        self._counter_widget.move_forward.connect(self._on_forward_pressed)
        min_height = self._counter_widget.minimumSizeHint().height()
        self._counter_widget.setFixedHeight(min_height)

        # setup the left pane
        left_layout = qtw.QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.addWidget(self._counter_widget)
        left_layout.addWidget(self._segmenter_pane)
        self._options_pane = qtw.QWidget()
        self._options_pane.setLayout(left_layout)

        # setup the main layout
        layout = qtw.QHBoxLayout()
        layout.addWidget(self._options_pane)
        layout.addWidget(right_widget)
        self.setLayout(layout)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(qtw.QSizePolicy.MinimumExpanding)
        policy.setVerticalPolicy(qtw.QSizePolicy.MinimumExpanding)
        self.setSizePolicy(policy)

        # setup the frames and the segmenters
        self._segmenters = {}
        self.set_frames(np.zeros((1, 160, 90), dtype=np.uint8))

        # add at least one segmenter
        if segmenters is None:
            self.add_segmenters(EllipseSegmenter("SEGMENTER1"))
        else:
            self.add_segmenters(segmenters)

        # make the segmenter pane width fixed
        width = self._options_pane.minimumSizeHint().width()
        self._options_pane.setFixedWidth(width)

        # set the qtimer
        self._timer = qtc.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.save_masks)

    # ****** SETTERS ****** #

    def set_frames(self, frames: NDArray):
        """
        set the frames to be labelled.

        Parameters
        ----------
        frames (NDArray):
            a 3D array with shape (frames, height, width) or a 4D
            array with shape (frames, height, width, channels).
        """

        # set the new frames
        utils.check_type(frames, np.ndarray)
        msg = "frames must be a 3D/4D numpy.ndarray with shape:\n"
        msg += " - 3D -> (frame, rows, cols)\n"
        msg += " - 4D -> (frame, rows, cols, rgba)."
        if frames.ndim not in (3, 4):
            raise ValueError(msg)
        self._frames = frames

        # update the counter widget
        self._counter_widget.set_max_counter(frames.shape[0])
        self._counter_widget.set_counter(1)

        # reset the segmenters
        self._segmenters = {}
        for widget in self._segmenter_pane.segmenters:
            sgm = widget.segmenter
            self._segmenters[sgm.name] = [sgm.copy()]

        # update the view
        self._update_view()
        self._update_saving()

    def set_formats(
        self,
        formats: dict[str, Callable],
    ):
        """
        set the file browsing file formats.

        Parameters
        ----------
        formats: dict[str, Callable]
            each key must represents a specific file format supported by the
            Labeller. Each key must correspond to a function accepting
            a string as input (the file selected by the file browser)
            and providing a 3D array with shape (frames, height, width) or a 4D
            array with shape (frames, height, width, channels) as output.
        """
        utils.check_type(formats, dict)
        self._formats = {}
        for key, fun in formats.items():
            utils.check_type(fun, Callable)  # type: ignore
            self._formats[key] = fun

    def add_segmenters(self, sgmntr: Segmenter | list[Segmenter]):
        """
        add a new segmenter to the list of available segmenters.

        Parameters
        ----------
        sgmntr: Segmenter | Iterable[Segmenter]
            the segmenter(s) to be included.
        """
        if isinstance(sgmntr, Segmenter):
            self._segmenter_pane.add_segmenter(sgmntr)
        elif isinstance(sgmntr, list):
            for sgm in sgmntr:
                self._segmenter_pane.add_segmenter(sgm)
        else:
            types = Segmenter | list[Segmenter]
            raise TypeError(f"sgmntr must be a any of {types}")

    def set_coords(self, coords: tuple[int, int] | None):
        """update the data coordinate label"""
        frm = "({}, {}, {})"
        if coords is None:
            self._data_coords.setText(frm.format(" ", " ", " "))
        else:
            vals = [str(i) for i in coords]
            vals += [str(self.frames[self.current_frame][coords[::-1]])]
            txt = frm.format(*vals)
            self._data_coords.setText(txt)

    def set_input(self, path: str):
        """
        set the file to be segmented.

        Parameters
        ----------
        path : str
            the path to the file to be automatically loaded once the labeller
            is opened.

        Raises
        ------
        TypeError
            if path is not a str instance or the string does not match
            with the path to an existing file.
        """

        # load the file on path
        if not isinstance(path, str) or not os.path.exists(path):
            raise TypeError("path must be an existing file.")
        self._input_bar._textfield.setText(path)

    # ****** GETTERS ****** #

    @property
    def segmenters(self):
        """return the list of stored segmenters"""
        return self._segmenters

    @property
    def frames(self):
        """return the list of images."""
        return self._frames

    @property
    def formats(self):
        """return the valid file formats."""
        return self._formats

    @property
    def available_frames(self):
        """return the number of available images."""
        return self._counter_widget.max_counter - 1

    @property
    def current_frame(self):
        """get index of the actual image shown."""
        return self._counter_widget.counter - 1

    @property
    def active_segmenter(self):
        """return the current active segmenter."""
        active = self._segmenter_pane.active_segmenter
        if active is None:
            return None
        return self.segmenters[active.name][self.current_frame]

    @property
    def current_segmenters(self):
        """return the segmenters linked to the current frame."""
        if len(self.segmenters) == 0:
            return {}
        return {i: v[self.current_frame] for i, v in self.segmenters.items()}

    @property
    def labels(self):
        """
        return the labels and the index of the segmentation_mask array
        corresponding to the given label.

        Example:

            segmentation_mask[:, :, :, labels["name"]]

            will return the array
            containing all the segmentation mask corresponding to the required
            label.
        """
        if self.segmenters is None:
            return {}
        return {v: i for i, v in enumerate(self.segmenters)}

    @property
    def segmentation_mask(self):
        """
        return a 4D numpy array containing the segmentation mask resulting
        from each frame and containing the boolean mask corresponding to each
        segmenter in the last dimension.
        """
        if self.segmenters is None or self.frames is None:
            return None
        shape = self.frames.shape[:3] + (len(self.segmenters),)
        masks = np.zeros(shape, dtype=bool)
        for i, frame in enumerate(self.frames):
            for j, name in enumerate(self.segmenters):
                if len(self.segmenters[name]) > i:
                    segmenter = self.segmenters[name][i]
                    fill, borders = segmenter.mask(frame.shape[:2])
                    if fill is not None and borders is not None:
                        masks[i, :, :, j] = fill | borders
        return masks

    # ****** EVENT HANDLERS ****** #

    def _on_forward_pressed(self):
        """handle the clicking of the forward button."""
        self._update_segmenters()
        self._update_view()
        self._update_saving()

    def _on_backward_pressed(self):
        """handle the clicking of the backward button."""
        self._update_view()

    def _on_input_text_changed(self, text: str):
        """handle the change in the file browse textfield."""

        # ensure that the current file has the appropriate file format.
        utils.check_type(text, str)
        ext = text.rsplit(".", maxsplit=1)[-1]
        if ext not in list(self.formats.keys()):
            qtw.QMessageBox.warning(
                self,
                "Invalid format",
                f"{self.text} has an invalid file format.",
            )

        # extract the new frames
        self.set_frames(self.formats[ext](text))

    def _on_save_pressed(self):
        """handle the press of the save button"""
        self._timer.start()

    def _on_save_completed(self):
        """inform about the end of the saving procedure"""
        msg = "Saving procedure complete."
        qtw.QMessageBox.information(self, "Saving complete", msg)

    def _on_added_segmenter(self, segmenter: SegmenterWidget):
        """handle the inclusion of a novel segmenter."""
        # add event handlers to the segmenter
        segmenter.name_changed.connect(self._on_name_changed)
        segmenter.color_changed.connect(self._on_color_changed)
        segmenter.linewidth_changed.connect(self._on_linewidth_changed)
        segmenter.fontsize_changed.connect(self._on_fontsize_changed)
        segmenter.text_checked.connect(self._on_widget_text_checked)

        # add the segmenting list and update
        self.segmenters[segmenter.segmenter.name] = [segmenter.segmenter.copy()]
        self._update_segmenters()
        self._update_view()

    def _on_removed_segmenter(self, segmenter: SegmenterWidget):
        """handle the removal of a segmenter."""
        removed = self.segmenters.pop(segmenter.segmenter.name)
        del removed
        self._update_segmenters()
        self._update_view()

    def _on_name_changed(
        self,
        source: SegmenterWidget,
        old: str,
        new: str,
    ):
        """handle the renaming action on one segmenter."""
        if old in [i for i in self.segmenters]:
            self.segmenters[new] = self.segmenters[old].copy()
            self.segmenters.pop(old)
        else:
            self.segmenters[new] = [source.segment.copy()]
        for segmenter in self.segmenters[new]:
            segmenter.set_name(new)
        self._update_segmenters()
        self._update_view()

    def _on_color_changed(
        self,
        source: SegmenterWidget,
        new: tuple[int, int, int, int],
    ):
        """handle the change in color of one segmenter."""
        for i in self.segmenters[source.segmenter.name]:
            i.set_color(new)
        self._update_view()

    def _on_fontsize_changed(
        self,
        source: SegmenterWidget,
        new: float,
    ):
        """handle the change in fontsize of one segmenter."""
        for i in self.segmenters[source.segmenter.name]:
            i.set_fontsize(int(new))
        self._update_view()

    def _on_linewidth_changed(
        self,
        source: SegmenterWidget,
        new: float,
    ):
        """handle the change in linewidth of one segmenter."""
        for i in self.segmenters[source.segmenter.name]:
            i.set_linewidth(int(new))
        self._update_view()

    def _on_widget_text_checked(self, source: SegmenterWidget):
        """handle the check/uncheck of the segmenter widgets text checkbox."""
        self._update_view()

    def _on_enter(self, coords: tuple[int, int] | None):
        """handle the mouse entering action"""
        self.set_coords(coords)

    def _on_leave(self, coords: tuple[int, int] | None):
        """handle the mouse leaving action"""
        self.set_coords(coords)

    def _on_press(self, coords: tuple[int, int] | list[int]):
        """handle the mouse clicking action"""
        # activate the clicked segmenter
        wdgts = self._segmenter_pane.segmenters[::-1]
        for key, segmenter in self.current_segmenters.items():
            if segmenter.isin(coords):
                widget = [i for i in wdgts if i.segmenter.name == key][0]
                if not widget.is_checked():
                    widget.click()
                    break

        # edit the active segmenter
        segmenter = self.active_segmenter
        if segmenter is not None:
            self._pressed = True
            if len(segmenter.points) == 0:
                segmenter.add_point(coords)  # type: ignore
            elif segmenter.is_drawable():
                segmenter.set_selected(segmenter.isin(coords))

    def _on_doubleclick(self, coords: tuple[int, int, Any] | None):
        """handle the mouse clicking action"""
        self._pressed = False
        segmenter = self.active_segmenter
        if segmenter is not None:
            segmenter.del_points()
            segmenter.set_selected(False)
            segmenter.set_angle(0)
            self._update_view()

    def _on_move(self, coords: tuple[int, int] | None):
        """handle the mouse motion action"""
        self.set_coords(coords)
        segmenter = self.active_segmenter
        if segmenter is not None and self._is_mouse_pressed():
            if segmenter.is_drawable():
                if segmenter.is_selected():
                    segmenter.shift(coords)  # type: ignore
                    self._update_view()
            else:
                segmenter.add_point(coords, 1)  # type: ignore
                self._update_view()
                segmenter.del_points(1)

    def _on_release(self, coords: tuple[int, int, Any] | None):
        """handle the mouse releasing action"""
        self._pressed = False
        segmenter = self.active_segmenter
        if segmenter is not None:
            if len(segmenter.points) == 1:
                segmenter.add_point(coords, 1)  # type: ignore
            self._update_view()
            segmenter.set_selected(False)

    def _on_wheeling(self, ticks: int):
        """handle the mouse wheel action"""
        self._pressed = False
        segmenter = self.active_segmenter
        if segmenter is not None and segmenter.is_drawable():
            segmenter.set_angle(segmenter.angle + ticks)
            self._update_view()

    # ****** METHODS ****** #

    def _estimate_new_segmenter(
        self,
        old_image: NDArray,
        new_image: NDArray,
        old_segmenter: Segmenter,
    ):
        """
        estimate the shift in the old_segmenter position according to the
        new image in order to maximize the the 2D cross-correlation.

        Paramters
        ---------
            old_image: (NDArray)
                the image on which the old_segmenter has been generated.

            new_image: (NDArray)
                the image on which the new_segmenter has to be shifted.

            old_segmenter: (Segmenter)
                the segmenter generated on old_image.

        Returns:
            new_segmenter (Segmenter):
                the shifted segmenter.
        """
        # check the inputs
        utils.check_type(old_image, np.ndarray)
        utils.check_type(new_image, np.ndarray)
        utils.check_type(old_segmenter, Segmenter)
        shape_ok = all(i == v for i, v in zip(old_image.shape, new_image.shape))
        assert shape_ok, "new_image and old_image must have the same shape."

        # check if the estimate can be obtained
        new_segmenter = old_segmenter.copy()
        if not old_segmenter.is_drawable():
            return new_segmenter

        # get the template
        fill, borders = old_segmenter.mask(old_image.shape[:2])[:2]
        ym, xm = np.where(fill | borders)
        xmin, xmax = np.min(xm), np.max(xm) + 1
        ymin, ymax = np.min(ym), np.max(ym) + 1
        old_mask = old_image[ymin:ymax, xmin:xmax]

        # convert to uint8
        def float2int(
            mask: NDArray,
            maxv: float,
            minv: float,
        ):
            return ((mask - minv) / (maxv - minv) * 255).astype(np.uint8)

        template = float2int(old_mask, np.max(old_mask), np.min(old_mask))

        # get the top-left corner of the segmenter bbox
        old_corner = np.array([np.min(xm), np.min(ym)])

        # get template matching method
        img = float2int(new_image, np.max(new_image), np.min(new_image))
        new_corner = []
        methods = [
            cv2.TM_CCOEFF,
            cv2.TM_CCORR,
            cv2.TM_SQDIFF,
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF_NORMED,
        ]
        for method in methods:
            res = cv2.matchTemplate(img, template, method)
            min_loc, max_loc = cv2.minMaxLoc(res)[-2:]
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left_corner = min_loc
            else:
                top_left_corner = max_loc
            new_corner += [np.atleast_2d(top_left_corner)]
        new_corner = np.median(np.vstack(new_corner), axis=0)
        delta = new_corner - old_corner

        # find the best match and update the position of the new segmenter
        new_center = (np.array(old_segmenter.center) + delta).astype(int)
        new_segmenter.shift(new_center.tolist())
        return new_segmenter

    def _is_mouse_pressed(self):
        """return whether the mouse is pressed on the figure."""
        return self._pressed

    def _update_view(self):
        """update the current artist to the actual counter."""
        if self.frames.shape[0] > 0:
            # get the image
            img = np.squeeze(self.frames[self.current_frame])

            # apply the correct colormap
            if img.ndim == 2:
                miv = np.min(img)
                mav = np.max(img) - miv
                if mav == 0:
                    nrm = np.zeros_like(img, dtype=np.uint8)
                else:
                    nrm = np.round(((img - miv) / mav) * 255).astype(np.uint8)
                cmap = cv2.applyColorMap(nrm, cv2.COLORMAP_VIRIDIS)
                alpha = np.ones(nrm.shape + (1,), dtype=np.uint8) * 255
                cmap = np.concatenate([cmap, alpha], axis=-1)
            else:
                cmap = img.astype(np.uint8)

            # apply the segmenters
            for segmenter in self.current_segmenters.values():
                if segmenter.is_drawable():
                    objs = segmenter.overlay(cmap)
                    if objs[0] is not None:
                        wdgts = self._segmenter_pane.segmenters
                        name = segmenter.name
                        widget = [i for i in wdgts if i.segmenter.name == name]
                        widget = widget[0]
                        if not widget.is_text_enabled():
                            objs = objs[:-1]
                        for mask in objs:
                            idx = np.where(mask > 0)
                            cmap[idx] = mask[idx]

            # update the view
            self._image_widget.set_ndarray(cmap)

    def _update_segmenters(self):
        """update the segmenters map to the actual state of the labeller."""
        for val in self.segmenters.values():
            while len(val) <= self.current_frame:
                if self._estimator_checkbox.isChecked():
                    new_segmenter = self._estimate_new_segmenter(
                        old_image=self.frames[self.current_frame - 1],
                        new_image=self.frames[self.current_frame],
                        old_segmenter=val[-1],
                    )
                else:
                    new_segmenter = val[-1].copy()
                val += [new_segmenter]

    def _update_saving(self):
        """update the save button."""
        state = self.current_frame == self.available_frames
        self._save_bar.setEnabled(state)

    def save_masks(self):
        """saving function used by QTimer"""

        # get the saving file
        file = qtw.QFileDialog.getSaveFileName(
            self,
            "Save File",
            self._last_path,
            "File Format (*.h5)",
        )
        if file is None:
            self._save_bar.reset()
        else:
            # update the path and the file
            file = file[0].replace("/", os.path.sep)
            self._last_path = file.rsplit(os.path.sep, 1)[0]

            # get the labels
            objs = {}
            labels = self.labels
            objs["labels"] = list(labels.keys())
            objs["indices"] = list(labels.values())
            self._save_bar.update()

            # get the masks
            objs["masks"] = self.segmentation_mask
            self._save_bar.update()

            # store the data
            obj = h5py.File(file, "w")
            for key, value in objs.items():
                obj.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                    compression_opts=9,
                )
                self._save_bar.update()
            obj.close()
            self._save_bar.update()

        # stop the timer
        self._timer.stop()
