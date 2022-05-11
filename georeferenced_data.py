import dataclasses
import typing

from osgeo import osr

from geotransformations import pix_to_geo
from scope_utils import Scope, Extent

_T = typing.TypeVar("_T")


@dataclasses.dataclass
class WithSrs(object):
    srs: typing.Union[str, osr.SpatialReference]


@dataclasses.dataclass(init=False)
class GeoreferencedData(WithSrs, typing.Generic[_T]):
    data: _T

    def __init__(self, data: _T, srs: typing.Union[str, osr.SpatialReference]) -> None:
        super().__init__(srs)
        self.data = data


@dataclasses.dataclass(init=False)
class WithScope(object):

    def __init__(self,
                 size: typing.Tuple[int, int],
                 geotransform: typing.Tuple[float, float, float, float, float, float],
                 scope: typing.Optional[Scope] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._size = size
        self._geotransform = geotransform
        self._scope = scope

        if scope is not None:
            x_off_geo, y_off_geo = pix_to_geo(scope.x_off, scope.y_off, geotransform)

            self._scoped_geotransform = (x_off_geo, geotransform[1], geotransform[2],
                                         y_off_geo, geotransform[4], geotransform[5])

    @property
    def size(self) -> typing.Tuple[int, int]:
        return self._size if not self.is_scoped else (self.scope.x_size, self.scope.y_size)

    @property
    def original_size(self) -> typing.Tuple[int, int]:
        return self._size

    @property
    def geotransform(self) -> typing.Tuple[float, float, float, float, float, float]:
        return self._geotransform if not self.is_scoped else self._scoped_geotransform

    @property
    def original_geotransform(self) -> typing.Tuple[float, float, float, float, float, float]:
        return self._geotransform

    @property
    def extent(self) -> Extent:
        x1, y1 = pix_to_geo(0, 0, self.geotransform)
        x2, y2 = pix_to_geo(self.size[0], self.size[1], self.geotransform)

        return Extent(min(x1, x2), min(y1, y2),
                      max(x1, x2), max(y1, y2))

    @property
    def original_extent(self) -> Extent:
        x1, y1 = pix_to_geo(0, 0, self.original_geotransform)
        x2, y2 = pix_to_geo(self.original_size[0], self.original_size[1], self.original_geotransform)

        return Extent(min(x1, x2), min(y1, y2),
                      max(x1, x2), max(y1, y2))

    @property
    def scope(self) -> typing.Optional[Scope]:
        return self._scope

    @property
    def is_scoped(self) -> bool:
        return self._scope is not None


@dataclasses.dataclass(init=False)
class RasterScope(WithScope,WithSrs):
    pass


_T2 = typing.TypeVar('_T2')


@dataclasses.dataclass(init=False)
class GeoreferencedRaster(WithScope, GeoreferencedData[_T]):

    def __init__(self, data: _T,
                 srs: typing.Union[str, osr.SpatialReference], *args, **kwargs):
        super(GeoreferencedRaster, self).__init__(data=data, srs=srs, *args, **kwargs)

    def copy_reference(self,
                       dst_class: typing.Type[_T2],
                       data: _T,
                       *args, **kwargs) -> _T2:
        return dst_class(data=data, srs=self.srs, size=self._size, geotransform=self._geotransform, scope=self._scope,
                         *args, **kwargs)
