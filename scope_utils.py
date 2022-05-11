import dataclasses
import typing
from dataclasses import dataclass

import numpy as np

from geotransformations import geo_to_pix


@dataclass(init=False, repr=False)
class Scope(object):

    def __init__(self,
                 x_off: int,
                 y_off: int,
                 x_size: int,
                 y_size: int,
                 x_min_border: int = 0,
                 x_max_border: int = 0,
                 y_min_border: int = 0,
                 y_max_border: int = 0):
        self._x_off = x_off
        self._y_off = y_off
        self._x_size = x_size
        self._y_size = y_size
        self._x_min_border = x_min_border
        self._x_max_border = x_max_border
        self._y_min_border = y_min_border
        self._y_max_border = y_max_border

    @property
    def x_off(self) -> int:
        return self._x_off - self._x_min_border

    @property
    def y_off(self) -> int:
        return self._y_off - self._y_min_border

    @property
    def x_size(self) -> int:
        return self._x_size + self._x_min_border + self._x_max_border

    @property
    def y_size(self) -> int:
        return self._y_size + self._y_min_border + self._y_max_border

    @property
    def x_min_border(self) -> int:
        return self._x_min_border

    @property
    def x_max_border(self) -> int:
        return self._x_max_border

    @property
    def y_min_border(self) -> int:
        return self._y_min_border

    @property
    def y_max_border(self) -> int:
        return self._y_max_border

    def without_borders(self) -> 'Scope':
        return Scope(x_off=self._x_off,
                     y_off=self._y_off,
                     x_size=self._x_size,
                     y_size=self._y_size)

    def with_merged_borders(self) -> 'Scope':
        return Scope(x_off=self.x_off,
                     y_off=self.y_off,
                     x_size=self.x_size,
                     y_size=self.y_size)

    def __repr__(self):
        including_borders = ''
        if self._x_min_border != 0 or self._x_max_border != 0 or \
                self._y_min_border != 0 or self._y_max_border != 0:
            including_borders = " including borders(x_min=%d, x_max=%d, y_min=%d, y_max=%d)" % (self._x_min_border,
                                                                                                self._x_max_border,
                                                                                                self._y_min_border,
                                                                                                self._y_max_border)
        return "Scope(x_off=%d, y_off=%d, x_size=%d, y_size=%d%s)" % (self.x_off,
                                                                      self.y_off,
                                                                      self.x_size,
                                                                      self.y_size,
                                                                      including_borders)


@dataclasses.dataclass
class Extent(object):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def intersect(self, other: 'Extent') -> 'Extent':
        return Extent(max(self.x_min, other.x_min),
                      max(self.y_min, other.y_min),
                      min(self.x_max, other.x_max),
                      min(self.y_max, other.y_max))

    def subextents(self, width: float, height: float) -> typing.Iterator['Extent']:

        base_y = self.y_min
        while base_y < self.y_max:
            base_x = self.x_min
            while base_x < self.x_max:
                x_max = min(base_x + width, self.x_max)
                y_max = min(base_y + height, self.y_max)
                yield Extent(base_x, base_y, x_max, y_max)

                base_x += width
            base_y += height

    def to_scope(self, geotransform: typing.Tuple[float, float, float, float, float, float],
                 raster_size: typing.Optional[typing.Tuple[int, int]] = None,
                 border: int = 0) -> Scope:
        cal_x, cal_y = (np.array([self.x_min, self.x_max]), np.array([self.y_min, self.y_max]))
        cal_x, cal_y = geo_to_pix(cal_x, cal_y, geotransform)

        x_off = max(int(np.floor(min(cal_x))), 0)
        y_off = max(int(np.floor(min(cal_y))), 0)

        x_max = int(np.ceil(max(cal_x)))
        y_max = int(np.ceil(max(cal_y)))

        if raster_size is not None:
            x_max = min(x_max, raster_size[0])
            y_max = min(y_max, raster_size[1])

        x_size = x_max - x_off
        y_size = y_max - y_off

        x_min_border = min(border, x_off)
        y_min_border = min(border, y_off)

        x_max_border = border
        y_max_border = border

        if raster_size is not None:
            x_max_border = min(x_max_border, raster_size[0] - (x_size + x_off))
            y_max_border = min(y_max_border, raster_size[1] - (y_size + y_off))

        return Scope(x_off=x_off,
                     y_off=y_off,
                     x_size=x_size,
                     y_size=y_size,
                     x_min_border=x_min_border,
                     y_min_border=y_min_border,
                     x_max_border=x_max_border,
                     y_max_border=y_max_border)
