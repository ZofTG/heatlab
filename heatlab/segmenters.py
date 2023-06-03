"""SEGMENTERS MODULE"""


#! IMPORTS

from typing import Iterable
from numpy.typing import NDArray

import cv2
import numpy as np

from .utils import Signal, check_type


__all__ = [
    "Segmenter",
    "RectangleSegmenter",
    "EllipseSegmenter",
    "TriangleSegmenter",
]

#! CLASSES


class Segmenter:
    """
    Generate a segmentation label via pyqt.

    Parameters
    ----------
    name: str
        the name of the label

    color: tuple[int, int, int, int]
        an RGBA color to be used for the patch.

    linewidth: int
        the linewidth of the object renderer.

    fontsize: int
        the text fontsize.
    """

    # ****** SIGNALS ****** #

    _name_changed: Signal
    _color_changed: Signal
    _linewidth_changed: Signal
    _fontsize_changed: Signal
    _angle_changed: Signal
    _points_changed: Signal
    _selected_changed: Signal

    # ****** VARIABLES ****** #

    _points: list = []
    _angle: float = 0.0
    _linewidth: int = 1
    _name: str = ""
    _color: tuple[int, int, int, int] = (0, 0, 0, 0)
    _fontsize: int = 4
    _selected: bool = False

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        name: str,
        color: tuple[int, int, int, int] | None,
        linewidth: int = 3,
        fontsize: int = 4,
    ):
        self._name_changed = Signal()
        self._color_changed = Signal()
        self._linewidth_changed = Signal()
        self._fontsize_changed = Signal()
        self._angle_changed = Signal()
        self._points_changed = Signal()
        self._selected_changed = Signal()
        self._points = []
        self.set_name(name)
        if color is None:
            color = tuple(np.random.randint(0, 256) for _ in range(3)) + (255,)
        self.set_color(color)
        self.set_linewidth(linewidth)
        self.set_fontsize(fontsize)
        self.set_angle(0)

    # ****** OPERATORS ****** #

    def __eq__(self, obj: object):
        """check equality between self and the provided object"""
        if isinstance(obj, Segmenter):
            name_eq = obj.name == self.name
            color_eq = all([i == v for i, v in zip(obj.color, self.color)])  # type: ignore
            class_eq = obj.__class__ == self.__class__
            return name_eq and color_eq and class_eq
        return False

    def __ne__(self, obj: object):
        """check inequality between self and the provided object"""
        return not self == obj

    # ****** SETTERS ****** #

    def set_angle(self, angle: float | int):
        """set the rotation angle in degrees"""

        # set the angle
        check_type(angle, (int, float))
        old = self._angle
        self._angle = float(angle)

        # emit the angle_changed signal
        self.angle_changed.emit(self, old, self._angle)

    def set_fontsize(self, fontsize: int):
        """set the fontsize"""
        check_type(fontsize, int)
        old = self._fontsize
        self._fontsize = fontsize
        self.fontsize_changed.emit(self, old, self._fontsize)

    def set_linewidth(self, linewidth: int):
        """set the linewidth of the segmenter."""
        check_type(linewidth, int)
        old = self._linewidth
        self._linewidth = linewidth
        self.linewidth_changed.emit(self, old, self._linewidth)

    def add_point(
        self,
        pnt: tuple[float | int, float | int] | list[float | int],
        index: int | None = None,
    ):
        """
        add a point to the segmenter list

        Parameters
        ----------
        pnt: tuple[float | int]
            the point to be added provided as tuple-like object with 2 elements.

        index: int | None
            the position of the point in the list.
        """

        # check the input
        if len(self.points) >= 2:  # type: ignore
            msg = "There are already 2 points stored on this segmenter."
            raise AttributeError(msg)
        check_type(pnt, Iterable)
        assert len(pnt) == 2, "pnt0 must have len = 2."
        for i in pnt:
            check_type(i, (int, float))
        if index is None:
            index = len(self.points)
        else:
            check_type(index, int)

        # add the point
        self._points.insert(index, pnt)
        self.points_changed.emit(self, self.points)

    def set_name(self, name: str):
        """set the label name."""
        check_type(name, str)
        old = self._name
        self._name = name
        self.name_changed.emit(self, old, name)

    def set_color(self, color: tuple[int, int, int, int]):
        """set the label rgb color."""

        # prepare the color
        check_type(color, (tuple, list))
        msg = "color must be a len=4 tuple of int in the 0-255 range."
        if len(color) != 4:
            raise ValueError(msg)
        for i in color:  # type: ignore
            check_type(i, int)

        # update
        old = self._color
        self._color = color
        self.color_changed.emit(self, old, color)

    def set_selected(self, state: bool):
        """
        set the state of the segmenter

        Parameters
        ----------
        state: bool
            set the selection state of the Segmenter to the provided state.
        """
        check_type(state, bool)
        old = self._selected
        self._selected = state
        if old is None or old != state:
            self._selected_changed.emit()

    # ****** PROPERTIES ****** #

    @property
    def angle_changed(self):
        """return the signal handling the change of the segmenter angle."""
        return self._angle_changed

    @property
    def fontsize_changed(self):
        """return the signal handling the change of the segmenter fontsize."""
        return self._fontsize_changed

    @property
    def name_changed(self):
        """return the signal handling the change of the segmenter name."""
        return self._name_changed

    @property
    def color_changed(self):
        """return the signal handling the change of the segmenter color."""
        return self._color_changed

    @property
    def linewidth_changed(self):
        """return the signal handling the change of the segmenter line width."""
        return self._linewidth_changed

    @property
    def points_changed(self):
        """return the signal handling the change of the segmenter points."""
        return self._points_changed

    @property
    def name(self):
        """return the label name."""
        return self._name

    @property
    def color(self):
        """return the label color"""
        return self._color

    @property
    def angle(self):
        """return the rotation angle in radians."""
        return self._angle

    @property
    def linewidth(self):
        """return the segmenter line width."""
        return self._linewidth

    @property
    def fontsize(self):
        """return the segmenter fontsize."""
        return self._fontsize

    @property
    def points(self):
        """return the points stored within the segmenter."""
        return self._points

    @property
    def center(self):
        """return the coordinates of the rectangle mid-point."""
        if len(self.points) == 2:
            pnt1, pnt2 = (np.array(i) for i in self.points[:2])
            x, y = (float(i) for i in ((pnt2 - pnt1) / 2 + pnt1))
            return x, y
        return 0.0, 0.0

    @property
    def height(self):
        """return the height of the box containing the segmenter."""
        if len(self.points) == 2:
            return float(abs(self.points[1][1] - self.points[0][1]))
        return 0.0

    @property
    def width(self):
        """return the width of the box containing the segmenter."""
        if len(self.points) == 2:
            return float(abs(self.points[1][0] - self.points[0][0]))
        return 0.0

    @property
    def bbox_corners(self):
        """return the corners of the bounding box containing the segmenter."""
        if len(self.points) != 2:
            return None

        # get the sorted coordinates of the delimiting points
        x0, y0 = self.points[0]
        x1, y1 = self.points[1]
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        # get the box corners
        cnt = self.center
        pBL = self.rotate2d((x0, y0), cnt, self.angle)
        pTL = self.rotate2d((x0, y1), cnt, self.angle)
        pBR = self.rotate2d((x1, y0), cnt, self.angle)
        pTR = self.rotate2d((x1, y1), cnt, self.angle)

        return pBL, pTL, pTR, pBR

    @property
    def bbox_domain(self):
        """return the domain of the bounding box containing the segmenter."""
        if len(self.points) != 2:
            return int(0), int(0)
        x_range = np.array(self.bbox_corners)[:, 0]
        return int(np.min(x_range)), int(np.max(x_range))

    @property
    def bbox_codomain(self):
        """return the codomain of the bounding box containing the segmenter."""
        if len(self.points) != 2:
            return int(0), int(0)
        y_range = np.array(self.bbox_corners)[:, 1]
        return int(np.min(y_range)), int(np.max(y_range))

    # ****** METHODS ****** #

    def is_selected(self):
        """return the state of the segmenter."""
        return self._selected

    def is_drawable(self):
        """check whether the current segmenter can be drawn."""
        return len(self.points) == 2

    def isin(
        self,
        coords: tuple[int, int] | list[int],
    ):
        """
        check whether the given coordinates are within the segmenter

        Parameters
        ----------
        coords: Iterable[int]
            the (x, y, value) coordinates of the point to be checked.

        Returns
        -------
        response: bool
            True if the coords are included in the segmenter. False, otherwise.
        """
        # check the inputs
        check_type(coords, Iterable)
        assert len(coords) == 2, "coords must have len = 2"
        for i in coords:
            check_type(i, int)

        # control if the coords point to an element within the object's mask
        if not self.is_drawable():
            return False
        y_max = int(max(self.bbox_codomain[1], coords[1]) + 1)
        x_max = int(max(self.bbox_domain[1], coords[0]) + 1)
        fill, borders = self.mask((y_max, x_max))
        if np.prod(fill.shape) == 0:
            return False
        if np.sum(fill.astype(int)) == 0 or np.sum(borders.astype(int)) == 0:
            return False
        return bool((fill | borders)[coords[1], coords[0]])

    def del_points(
        self,
        index: int | None = None,
    ):
        """
        remove points from the segmenter

        Parameters
        ----------
        index: int | None
            the index of the point to be removed. If None, all the points
            are removed.
        """
        if index is None:
            self._points = []
        else:
            check_type(index, int)
            self._points.pop(index)
        self.points_changed.emit(self, self.points)

    def copy(self):
        """return a deep copy of the object."""
        cls = self.__class__
        obj = cls(
            name=self.name,
            color=self.color,
            linewidth=self.linewidth,
            fontsize=self.fontsize,
        )
        for pnt in self.points:
            obj.add_point(pnt)
        obj.set_angle(self.angle)
        return obj

    def shift(
        self,
        pnt: tuple[float | int, float | int] | list[float | int],
    ):
        """
        shift the patch by the given coordinates

        Parameters
        ----------
        pnt: Iterable[int]
            the reference point to be used as reference to shift the segmenter
            on the image.
        ."""
        # check the input
        check_type(pnt, Iterable)
        assert len(pnt) == 2, "obj must have len = 2."
        for i in pnt:
            check_type(i, int)

        # apply the offset
        off = tuple(i - v for i, v in zip(pnt, self.center))
        for p, pnt in enumerate(self._points):
            self._points[p] = tuple(i + v for i, v in zip(pnt, off))

    def rotate2d(
        self,
        pnt: tuple[float | int, float | int],
        cnt: tuple[float | int, float | int],
        angle: float | int,
    ):
        """
        rotate the given point according to the segmenter angle with
        respect to its centre.

        Parameters
        ----------
        pnt: tuple[float | int, float | int]
            the point to be rotated in (x, y) coordinates.

        cnt: tuple[float | int, float | int] | None
            the point being the centre of the rotation.

        angle: float | int
            the rotation angle in degrees

        Returns
        -------
        rot: tuple[float, float]
            the rotated point.
        """
        # check the input
        for i in (pnt, cnt):
            check_type(i, Iterable)
            assert len(pnt) == 2, "pnt and cnt must have len = 2"
            for i in pnt:
                check_type(i, int | float)
        check_type(angle, (float, int))

        # get the rotation matrix
        theta = angle / 180 * np.pi
        cos = np.cos(theta)
        sin = np.sin(theta)
        mat = np.array([[cos, -sin], [sin, cos]])

        # return the rotated point
        rot = np.atleast_2d(tuple(i - c for i, c in zip(pnt, cnt))).T
        rot = (mat @ rot).flatten()
        return tuple(i + c for i, c in zip(rot, cnt))

    def _fill_between(
        self,
        shape: tuple[int, int] | list[int],
        x: NDArray,
        y0: NDArray,
        y1: NDArray,
    ):
        """
        fill the space in image within the y0-y1 range along the x axis.

        Parameters
        ----------
        shape: tuple[int, int]
            the shape of the output image in [rows, cols].

        x: NDArray
            the x axis coordinates.

        y0: NDArray
            the first set of y axis coordinates.

        y1: NDArray
            the second set of y axis coordinates.

        Returns
        -------
        mask: NDArray
            a 2D array with the provided shape of bool values where
            True denotes the elements filled.
        """
        # check the inputs
        check_type(shape, Iterable)
        assert len(shape) == 2, "shape must have len = 2"
        for i in shape:
            check_type(i, int)
        check_type(x, np.ndarray)
        assert x.ndim == 1, "'x' must be a 1D array."
        check_type(y0, np.ndarray)
        assert y0.ndim == 1, "'y0' must be a 1D array."
        check_type(y1, np.ndarray)
        assert y1.ndim == 1, "'y1' must be a 1D array."
        assert len(x) == len(y0), "x, y0 and y1 must have the same len."
        assert len(x) == len(y1), "x, y0 and y1 must have the same len."

        # get the filled mask
        mat = np.vstack(np.atleast_2d(x, y0, y1))
        xs, y0, y1 = mat[:, ~np.any(np.isnan(mat), axis=0)]
        xs = xs.astype(int)
        mask_y = [np.tile(i, shape[1]) for i in np.arange(shape[0])]
        mask_y = np.atleast_2d(mask_y)
        mask = np.zeros(shape, dtype=bool)
        mask[:, xs] = (mask_y[:, xs] >= y0) & (mask_y[:, xs] <= y1)
        return mask.astype(bool)

    def icon(
        self,
        shape: tuple[int, int] | list[int] = (20, 20),  # type: ignore
        ratio: float | int = 16 / 9,
    ):
        """
        return a 2D numpy array of bool representing the mask drawing the
        segmenter.

        Parameters
        ----------
        shape: Iterable[int]
            a len=2 iterable defining the desired shape of the output icon in
            [row, col].

        ratio: float | int
            the aspect-ratio of the figure rendered within the figure.

        Returns
        -------
        icon: NDArray
            a 2D array with the given shape with the mask resulting by the
            object.
        """
        # check the inputs
        check_type(shape, Iterable)
        assert len(shape) == 2, "shape must be a len=2 Iterable object."
        for i in shape:
            check_type(i, int)
        check_type(ratio, (int, float))

        # generate the ellipse
        width = shape[1]
        height = width / ratio
        center = tuple(np.array(shape) / 2)[::-1]
        x = np.arange(width)
        y0, y1 = self._getY(x, center, width, height, 0)
        mask = self._fill_between(shape, x, y0, y1)
        clr = np.expand_dims(np.expand_dims(self.color, 0), 0)
        return (np.expand_dims(mask, 2) * clr).astype(np.uint8)

    def mask(
        self,
        shape: tuple[int, int] | list[int],
    ):
        """
        return the mask of the image.

        Parameters
        ----------
        shape: Iterable[int]
            a len=2 iterable defining the desired shape of the output icon in
            [row, col].

        Returns
        -------
        mask: NDArray
            a bool 2D mask indicating the pixels within the mask.
        """
        # check the input
        check_type(shape, tuple | list)
        assert len(shape) == 2, "shape must be a len=2 Iterable object."
        for i in shape:
            check_type(i, int)
        if (
            self.width == 0
            or self.height == 0
            or self.width is None
            or self.height is None
        ):
            return np.ndarray([]).astype(bool), np.ndarray([]).astype(bool)

        # get the whole mask
        domain = self.bbox_domain
        x0 = max(0, int(np.ceil(domain[0])))
        x1 = min(shape[1], int(np.floor(domain[1])))
        x = np.arange(x0, x1)
        ys = self._getY(x, self.center, self.width, self.height, self.angle)
        msk = self._fill_between(shape, x, ys[0], ys[1])
        codomain = self.bbox_codomain
        y = np.arange(shape[0])
        msk[(y < codomain[0]) & (y > codomain[1]), :] = False

        # obtain only the filling area
        w = self.width - self.linewidth * 2
        h = self.height - self.linewidth * 2
        ys = self._getY(x, self.center, w, h, self.angle)
        fill = self._fill_between(shape, x, ys[0], ys[1])
        borders = msk.astype(np.uint8) - fill.astype(np.uint8)
        return fill, borders.astype(bool)

    def overlay(
        self,
        image: NDArray,
    ):
        """overlay the current segmenter to the image."""

        # check the input
        check_type(image, np.ndarray)
        msg = "image must be a 3D numpy array with dtype=uint8."
        assert image.ndim == 3 and image.dtype == np.uint8, msg

        # get the filling and border color
        fill, borders = self.mask(image.shape[:2])
        if np.prod(fill.shape) == 0 or np.prod(borders.shape) == 0:
            out = np.zeros_like(image).astype(np.uint8)
            return out, out, out
        mask = fill | borders
        if not np.any(mask):
            out = np.zeros_like(image).astype(np.uint8)
            return out, out, out

        # get the overlaying masks
        fill_idx = np.where(fill)
        fill_clr = image[fill_idx].astype(float)
        fill_clr[:, :3] *= 0.5
        if fill_clr.ndim == 2:
            fill_clr[:, :3] += np.atleast_2d(self.color[:3]) * 0.5
        else:
            fill_clr[:, :3] += np.atleast_3d(self.color[:3]) * 0.5
        fill_clr = np.round(fill_clr).astype(np.uint8)
        fill_mask = np.zeros(image.shape, dtype=np.uint8)
        fill_mask[fill_idx] = fill_clr
        bord_idx = np.where(borders)
        bord_clr = image[bord_idx].astype(float)
        bord_clr[:, :3] *= 0.2
        if bord_clr.ndim == 2:
            bord_clr[:, :3] += np.atleast_2d(self.color[:3]) * 0.8
        else:
            bord_clr[:, :3] += np.atleast_3d(self.color[:3]) * 0.8
        bord_clr = np.round(bord_clr).astype(np.uint8)
        bord_mask = np.zeros(image.shape, dtype=np.uint8)
        bord_mask[bord_idx] = bord_clr

        # add the text
        if fill.ndim == 0:
            org = (0, 0)
            clr = (0, 0, 0, 0)
        else:
            org = tuple(np.min(i) for i in np.where(mask > 0))[::-1]
            clr = self.color[:3] + (255,)
        text_mask = cv2.putText(
            img=np.zeros_like(image, dtype=np.uint8),
            text=self.name,
            org=org,
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=self.fontsize * 0.1,
            color=clr,
            thickness=1,
            bottomLeftOrigin=False,
            lineType=cv2.FILLED,
        )
        return fill_mask, bord_mask, text_mask.astype(np.uint8)

    def _getY(
        self,
        x: NDArray,
        center: tuple[int | float, int | float],
        width: int | float,
        height: int | float,
        angle: int | float,
    ):
        """
        private method used to obtain the coordinates of the ellipsis

        Parameters
        ----------
        x: list-like array
            the x coordinates.

        center: tuple[int | float, int | float]
            the center of the ellipse.

        width: int | float
            the full width of the ellipse

        height: int | float
            the full height of the ellipse

        angle: int | float
            the rotation angle in degrees

        Returns
        -------
        y0: list-like array
            the first set of y coordinates being the boundaries of the
            ellipse.

        y1: list-like array
            the second set of y coordinates being the second boundary of the
            ellipse.
        """
        raise NotImplementedError


class RectangleSegmenter(Segmenter):
    """
    Generate a rectangle selector.

    Parameters
    ----------
    name: str
        the name of the label

    color: tuple[int, int, int, int]
        an rgba color to be used for the patch.

    linewidth: int | float
        the linewidth of the object renderer.

    fontsize: int
        the text fontsize.
    """

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        name: str,
        color: tuple[int, int, int, int] | None = None,
        linewidth: int = 3,
        fontsize: int = 4,
    ):
        super().__init__(name, color, linewidth, fontsize)

    # ****** METHODS ****** #

    def _get_y_values(
        self,
        x: NDArray,
        pnt0: tuple[float, float],
        pnt1: tuple[float, float],
    ):
        """
        get the slope and intercept of the line joining pnt0 and pnt1

        Parameters
        ----------
        x: NDArray
            the x axis values on which the line joining pnt0 and pnt1 is
            returned

        pnt0: tuple[float, float]
            the first point

        pnt1: tuple[float, float]
            the second point

        Results
        -------
        coefs: tuple[float, float]
            the slope and intercept of the line.
        """
        # check the inputs
        for p in (pnt0, pnt1):
            check_type(p, Iterable)
            assert len(p) == 2, "pnt0 and pnt1 must have len = 2"
            for i in p:
                check_type(i, int | float)
        check_type(x, np.ndarray)
        assert x.ndim == 1, "x must be a 1D array"

        # get the line
        if pnt0[0] == pnt1[0]:
            return np.tile(np.nan, len(x))
        slope = (pnt1[1] - pnt0[1]) / (pnt1[0] - pnt0[0])
        intercept = pnt1[1] - slope * pnt1[0]
        return x * slope + intercept

    def _getY(
        self,
        x: NDArray,
        center: tuple[int | float, int | float],
        width: int | float,
        height: int | float,
        angle: int | float,
    ):
        """
        get the y coordinates of the ellipsis given x.

        Parameters
        ----------
        x: list-like array
            the x coordinates.

        center: tuple[int | float, int | float]
            the center of the ellipse.

        width: int | float
            the full width of the ellipse

        height: int | float
            the full height of the ellipse

        angle: int | float
            the rotation angle in degrees

        Returns
        -------
        y0: list-like array
            the first set of y coordinates being the boundaries of the
            ellipse.

        y1: list-like array
            the second set of y coordinates being the second boundary of the
            ellipse.
        """
        # check the inputs
        check_type(x, np.ndarray)
        assert x.ndim == 1, "'x' must be a 1D array."
        check_type(center, Iterable)
        assert len(center) == 2, "center must have len = 2"
        for i in center:
            check_type(i, int | float)
        check_type(width, (int, float))
        check_type(height, (int, float))
        check_type(angle, (int, float))

        # get the corners of the rectangle
        px0, py0 = center
        wdh = width / 2
        hgt = height / 2
        ptA = self.rotate2d((px0 - wdh, py0 - hgt), center, angle)
        ptB = self.rotate2d((px0 - wdh, py0 + hgt), center, angle)
        ptC = self.rotate2d((px0 + wdh, py0 + hgt), center, angle)
        ptD = self.rotate2d((px0 + wdh, py0 - hgt), center, angle)

        # get the y values of each line joining the corners and fitted on x
        lAB = self._get_y_values(x, ptA, ptB)
        lBC = self._get_y_values(x, ptB, ptC)
        lCD = self._get_y_values(x, ptC, ptD)
        lDA = self._get_y_values(x, ptD, ptA)
        lines = np.vstack(np.atleast_2d([lAB, lBC, lCD, lDA]))  # type: ignore
        lines = lines[~np.all(np.isnan(lines), axis=1)]
        sorted_lines = np.sort(lines, axis=0)
        null_values = np.any(np.isnan(sorted_lines), axis=0)
        sorted_lines[:, null_values] = np.nan

        # for each x value, get the margins of the rectangle
        if sorted_lines.shape[0] < 2:
            y0 = np.tile(np.nan, len(x))
            y1 = np.tile(np.nan, len(x))
        elif sorted_lines.shape[0] == 2:
            y0, y1 = sorted_lines
        else:
            y0, y1 = sorted_lines[1:-1]
        return y0, y1


class EllipseSegmenter(Segmenter):
    """
    Generate an ellipse selector.

    Parameters
    ----------
    name: str
        the name of the label

    color: tuple[int, int, int, int]
        an rgba color to be used for the patch.

    linewidth: int
        the linewidth of the object renderer.

    fontsize: int
        the text fontsize.
    """

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        name: str,
        color: tuple[int, int, int, int] | None = None,
        linewidth: int = 3,
        fontsize: int = 4,
    ):
        super().__init__(name, color, linewidth, fontsize)

    # ****** METHODS ****** #

    def _getY(
        self,
        x: NDArray,
        center: tuple[int | float, int | float],
        width: int | float,
        height: int | float,
        angle: int | float,
    ):
        """
        get the y coordinates of the ellipsis given x.

        Parameters
        ----------
        x: list-like array
            the x coordinates.

        center: tuple[int | float, int | float]
            the center of the ellipse.

        width: int | float
            the full width of the ellipse

        height: int | float
            the full height of the ellipse

        angle: int | float
            the rotation angle in degrees

        Returns
        -------
        y0: list-like array
            the first set of y coordinates being the boundaries of the
            ellipse.

        y1: list-like array
            the second set of y coordinates being the second boundary of the
            ellipse.
        """
        # check the inputs
        check_type(x, np.ndarray)
        assert x.ndim == 1, "'x' must be a 1D array."
        check_type(center, Iterable)
        assert len(center) == 2, "center must have len = 2"
        for i in center:
            check_type(i, int | float)
        check_type(width, (int, float))
        check_type(height, (int, float))
        check_type(angle, (int, float))

        # get the ellipsis dimensions
        t = angle * np.pi / 180
        xc, yc = center
        sin2 = np.sin(t) ** 2
        cos2 = np.cos(t) ** 2
        b2 = (height * 0.5) ** 2
        a2 = (width * 0.5) ** 2

        # obtain the parametric coefficients
        A = b2 * cos2 + a2 * sin2
        B = (b2 - a2) * np.sin(2 * t)
        C = b2 * sin2 + a2 * cos2
        D = -(2 * A * xc + B * yc)
        E = -(2 * C * yc + B * xc)
        F = A * xc**2 + C * yc**2 + B * xc * yc - a2 * b2

        # now we derive the solution of the second order equation allowing
        # to find y from:
        #   Ax**2 + Bxy + Cy ** 2 + Dx + Ey + F = 0
        # which can be rewritten as:
        #   Cy ** 2 + (Bx + E)y + (Ax ** 2 + Dx + F) = 0
        # therefore, the solution will be:
        a = C
        b = B * x + E
        c = A * x**2 + D * x + F
        d = b**2 - 4 * a * c
        null_values = np.where(d < 0)[0]
        d[null_values] = 0
        e = 2 * a
        f = (d**0.5) / e
        g = -b / e
        y0 = g - f
        y1 = g + f
        y0[null_values] = np.nan
        y1[null_values] = np.nan
        return y0, y1


class TriangleSegmenter(RectangleSegmenter):
    """
    Generate a rectangle selector.

    Parameters
    ----------
    name: str
        the name of the label

    color: tuple[int, int, int, int]
        an rgba color to be used for the patch.

    linewidth: int | float
        the linewidth of the object renderer.

    fontsize: int
        the text fontsize.
    """

    # ****** CONSTRUCTOR ****** #

    def __init__(
        self,
        name: str,
        color: tuple[int, int, int, int] | None = None,
        linewidth: int = 3,
        fontsize: int = 4,
    ):
        super().__init__(name, color, linewidth, fontsize)

    # ****** METHODS ****** #

    def _getY(
        self,
        x: NDArray,
        center: tuple[int | float, int | float],
        width: int | float,
        height: int | float,
        angle: int | float,
    ):
        """
        get the y coordinates of the ellipsis given x.

        Parameters
        ----------
        x: list-like array
            the x coordinates.

        center: tuple[int | float, int | float]
            the center of the ellipse.

        width: int | float
            the full width of the ellipse

        height: int | float
            the full height of the ellipse

        angle: int | float
            the rotation angle in degrees

        Returns
        -------
        y0: list-like array
            the first set of y coordinates being the boundaries of the
            ellipse.

        y1: list-like array
            the second set of y coordinates being the second boundary of the
            ellipse.
        """
        # check the inputs
        check_type(x, np.ndarray)
        assert x.ndim == 1, "'x' must be a 1D array."
        check_type(center, Iterable)
        assert len(center) == 2, "center must have len = 2"
        for i in center:
            check_type(i, int | float)
        check_type(width, (int, float))
        check_type(height, (int, float))
        check_type(angle, (int, float))

        # get the base and the vertex of the rectangle
        w = width / 2
        h = height / 2
        x0, y0 = center
        ptA = self.rotate2d((x0 - w, y0 - h), center, angle + 180)
        pTL = self.rotate2d((x0 - w, y0 + h), center, angle + 180)
        pTR = self.rotate2d((x0 + w, y0 + h), center, angle + 180)
        ptB = self.rotate2d((x0 + w, y0 - h), center, angle + 180)
        ptC = tuple((i + v) * 0.5 for i, v in zip(pTL, pTR))

        # get the y values of each line joining the corners and fitted on x
        lAB = self._get_y_values(x, ptA, ptB)
        lBC = self._get_y_values(x, ptB, ptC)
        lAC = self._get_y_values(x, ptA, ptC)

        # get the domain and codomain
        pnts = np.vstack(np.atleast_2d(ptA, ptB, ptC))
        x_rng = (min(pnts[:, 0]), max(pnts[:, 0]))
        y_rng = (min(pnts[:, 1]), max(pnts[:, 1]))

        # get the lines
        lines = np.vstack(np.atleast_2d([lAB, lBC, lAC]))  # type: ignore
        lines[:, (x < x_rng[0]) | (x > x_rng[1])] = np.nan
        lines[(lines < y_rng[0]) | (lines > y_rng[1])] = np.nan
        sorted_lines = np.sort(lines, axis=0)

        return sorted_lines[0], sorted_lines[1]
