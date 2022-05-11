from dataclasses import dataclass
import numpy as np
import typing

from osgeo import osr, gdal, gdal_array

from georeferenced_data import GeoreferencedRaster
from scope_utils import Scope, Extent


@dataclass(init=False)
class BathymetryEstimations(GeoreferencedRaster[np.ndarray]):

    def __init__(self, data: np.ndarray, srs: typing.Union[str, osr.SpatialReference],
                 size: typing.Tuple[int, int],
                 geotransform: typing.Tuple[float, float, float, float, float, float],
                 scope: typing.Optional[Scope] = None,
                 date: typing.Optional[str] = None):
        super().__init__(data=data, srs=srs, size=size, geotransform=geotransform, scope=scope)
        self.date = date

    date: typing.Optional[str] = None


def load_bathymetry(path: str, extent: typing.Optional[Extent] = None, border: typing.Optional[int] = 0) -> BathymetryEstimations:

    dataset: gdal.Dataset = gdal.Open(path)

    if dataset is None:
        raise FileNotFoundError("Dataset \"%s\" cannot be opened" % path)

    ret = load_sentinel_from_dataset(dataset, extent, border)

    return ret


def load_sentinel_from_dataset(dataset: gdal.Dataset, extent: typing.Optional[Extent] = None, border: typing.Optional[int] = 0):

    geotransform: typing.Tuple[float, float, float, float, float, float] = dataset.GetGeoTransform()

    scope: typing.Optional[Scope] = None
    size: typing.Tuple[int, int] = (dataset.RasterXSize, dataset.RasterYSize)

    if extent is not None:
        print("Calculating custom raster scope")
        scope = extent.to_scope(geotransform,size, border)
        print("New scope is %s" % scope)

    raw_data: np.ndarray
    band: gdal.Band = dataset.GetRasterBand(1)

    if scope is None:
        raw_data = band.ReadAsArray()
    else:
        raw_data = gdal_array.BandReadAsArray(band,scope.x_off, scope.y_off,
                                                    scope.x_size, scope.y_size,
                                                    scope.x_size, scope.y_size)

    ret = BathymetryEstimations(data=raw_data, size=size, geotransform=geotransform,
                                srs=dataset.GetSpatialRef(), date=dataset.GetMetadataItem("DATE"),
                                scope=scope)

    return ret