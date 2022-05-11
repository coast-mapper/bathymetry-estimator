import typing
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from osgeo import gdal, gdal_array, osr

from georeferenced_data import GeoreferencedRaster, RasterScope
from scope_utils import Scope, Extent
from geotransformations import pix_to_geo, geo_to_pix
from models.abstract import input_data


@dataclass(init=False)
class SentinelData(GeoreferencedRaster[Dict[str, np.ndarray]]):

    def __init__(self, data: Dict[str, np.ndarray], srs: typing.Union[str, osr.SpatialReference],
                 size: typing.Tuple[int, int],
                 geotransform: typing.Tuple[float, float, float, float, float, float],
                 scope: typing.Optional[Scope] = None,
                 date: Optional[str] = None):
        super().__init__(data=data, srs=srs, size=size, geotransform=geotransform, scope=scope)
        self.date = date

    date: Optional[str] = None


def determine_raster_extent(path: str, extent: typing.Optional[Extent] = None) -> RasterScope:
    dataset: gdal.Dataset = gdal.Open(path)

    if dataset is None:
        raise FileNotFoundError("Dataset \"%s\" cannot be opened" % path)

    geotransform: Tuple[float, float, float, float, float, float] = dataset.GetGeoTransform()

    scope: typing.Optional[Scope] = None
    size: typing.Tuple[int, int] = (dataset.RasterXSize, dataset.RasterYSize)

    if extent is not None:
        print("Calculating custom raster scope")
        scope = extent.to_scope(geotransform, size)
        print("New extent is %s" % scope)



    return RasterScope(size=size, geotransform=geotransform,
                       srs=dataset.GetSpatialRef(),
                       scope=scope)


def load_sentinel(path: str, extent: typing.Optional[Extent] = None, border: typing.Optional[int] = 0) -> SentinelData:

    dataset: gdal.Dataset = gdal.Open(path)

    if dataset is None:
        raise FileNotFoundError("Dataset \"%s\" cannot be opened" % path)

    ret = load_sentinel_from_dataset(dataset, extent, border)

    return ret


def load_sentinel_from_dataset(dataset: gdal.Dataset, extent: typing.Optional[Extent] = None, border: typing.Optional[int] = 0):

    bands_to_find = [input_data.B2, input_data.B3, input_data.B4, input_data.B8]
    geotransform: Tuple[float, float, float, float, float, float] = dataset.GetGeoTransform()

    scope: typing.Optional[Scope] = None
    size: typing.Tuple[int, int] = (dataset.RasterXSize, dataset.RasterYSize)

    if extent is not None:
        print("Calculating custom raster scope")
        scope = extent.to_scope(geotransform,size, border)
        print("New scope is %s" % scope)

    raw_data: Dict[str, np.ndarray] = {}

    for band_no in range(1, dataset.RasterCount + 1):
        band: gdal.Band = dataset.GetRasterBand(band_no)
        name = band.GetMetadataItem("BANDNAME")
        if name is None:
            print("No band name for band %d" % band_no)
            continue

        if name in bands_to_find:
            if scope is None:
                raw_data[name] = band.ReadAsArray() / 10000
            else:
                raw_data[name] = gdal_array.BandReadAsArray(band,
                                                            scope.x_off, scope.y_off,
                                                            scope.x_size, scope.y_size,
                                                            scope.x_size, scope.y_size) / 10000

    not_present = set(bands_to_find) - set(raw_data.keys())

    if len(not_present) != 0:
        print("Following bands not found: %s" % not_present)
        raise RuntimeError("Bands not found: %s" % not_present)

    ret = SentinelData(data=raw_data, size=size, geotransform=geotransform,
                       srs=dataset.GetSpatialRef(), date=dataset.GetMetadataItem("PRODUCT_START_TIME"),
                       scope=scope)

    return ret


@dataclass(init=False)
class ModelInputData(GeoreferencedRaster[typing.Dict[str, np.ndarray]]):

    def __init__(self, data: typing.Dict[str, np.ndarray], srs: typing.Union[str, osr.SpatialReference],
                 size: typing.Tuple[int, int],
                 geotransform: typing.Tuple[float, float, float, float, float, float],
                 scope: typing.Optional[Scope],
                 mask: np.ndarray,
                 date: typing.Optional[str] = None):
        super().__init__(data=data, srs=srs, size=size, geotransform=geotransform, scope=scope)
        self.mask = mask
        self.date = date

    mask: np.ndarray
    date: Optional[str] = None


def prepare_input_data(data: SentinelData, required_input_data: typing.List[str],
                       perform_gauss_filtering: bool = False,
                       aux_mask: typing.Optional[np.ndarray] = None) -> ModelInputData:
    from scipy.ndimage import gaussian_filter

    mask1 = data.data[input_data.B8] < 0.15
    mask2 = (data.data[input_data.B2] + data.data[input_data.B3] + data.data[input_data.B4]) < 0.32

    mask3: np.ndarray = (data.data[input_data.B2] > 0) & (data.data[input_data.B3] > 0) & (data.data[input_data.B4] > 0) \
                        & (data.data[input_data.B8] > 0)

    mask4: np.ndarray = (data.data[input_data.B2] < 65.535) & (data.data[input_data.B3] < 65.535) & (data.data[input_data.B4] < 65.535) \
                        & (data.data[input_data.B8] < 65.535)

    sea_mask: np.ndarray = mask1 & mask2 & mask3 & mask4

    if aux_mask is not None:
        sea_mask = sea_mask & aux_mask

    if perform_gauss_filtering:
        float_mask = sea_mask.astype(np.float)
        filtered_mask = gaussian_filter(float_mask, 2)

    ret_dict: typing.Dict[str, np.ndarray] = {}

    if input_data.raw_bathymetry in required_input_data:
        bath_estimation: np.ndarray = np.log(data.data[input_data.B2] + 1) / np.log(data.data[input_data.B3] + 1)

        bath_estimation[sea_mask == False] = 0

        if perform_gauss_filtering:
            bath_estimation = gaussian_filter(bath_estimation, 2)
            bath_estimation = bath_estimation / filtered_mask

        ret_dict[input_data.raw_bathymetry] = bath_estimation

    bands = [input_data.B2, input_data.B3, input_data.B4, input_data.B8]

    for b in bands:
        if b in required_input_data:
            band_data = data.data[b]

            if perform_gauss_filtering:
                band_data = gaussian_filter(band_data, 2)
                band_data = band_data / filtered_mask
                band_data[band_data >= 65.535] = float('nan')

            ret_dict[b] = band_data

    if input_data.x in required_input_data or input_data.y in required_input_data:
        x_pix, y_pix = np.meshgrid(range(data.size[0]), range(data.size[1]))
        x_geo, y_geo = pix_to_geo(x_pix, y_pix, data.geotransform)

        if input_data.x in required_input_data:
            ret_dict[input_data.x] = x_geo

        if input_data.x in required_input_data:
            ret_dict[input_data.y] = y_geo

    ret = data.copy_reference(ModelInputData, data=ret_dict, date=data.date, mask=sea_mask)

    return ret
