from osgeo import gdal
import typing

_T = typing.TypeVar("_T")


def pix_to_geo(x: _T, y: _T, geotransform: typing.Tuple[float, float, float, float, float, float]) -> typing.Tuple[
    _T, _T]:
    x_geo = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    y_geo = geotransform[3] + x * geotransform[4] + y * geotransform[5]

    return x_geo, y_geo


def geo_to_pix(x, y, geotransform: typing.Tuple[float, float, float, float, float, float]) -> typing.Tuple[_T, _T]:
    inverse = gdal.InvGeoTransform(geotransform)

    x_pix = inverse[0] + x * inverse[1] + y * inverse[2]
    y_pix = inverse[3] + x * inverse[4] + y * inverse[5]

    return x_pix, y_pix
