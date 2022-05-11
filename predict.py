import argparse
import typing

import numpy as np
import pandas as pd
from osgeo import gdal_array

import argparse_def
# noinspection PyUnresolvedReferences
import matplotlib_setup
import models.abstract
import sentinel_data_utils
import reference_data_utils
from osgeo_setup import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse_def.define_input_raster_options(parser)
    argparse_def.define_prediction_specific_options(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    model = models.abstract.AbstractModel.load_model(args.model_dir)

    raster_scope = sentinel_data_utils.determine_raster_extent(args.sentinel_data, model.model_extent)

    sentinel_input: gdal.Dataset = gdal.Open(args.sentinel_data)

    if sentinel_input is None:
        raise FileNotFoundError("Dataset \"%s\" cannot be opened" % args.sentinel_data)

    driver: gdal.Driver = gdal.GetDriverByName('GTiff')

    res_size = raster_scope.original_size if args.preserve_original_size else raster_scope.size
    res_geotransform = raster_scope.original_geotransform if args.preserve_original_size else raster_scope.geotransform

    ds: gdal.Dataset = driver.Create(args.result_file, xsize=res_size[0],
                                         ysize=res_size[1],
                                         bands=1, eType=gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])

    band: gdal.Band = ds.GetRasterBand(1)
    band.Fill(float('nan'))

    mask_file: typing.Optional[gdal.Dataset] = None

    if args.mask_file:
        mask_file = gdal.Open(args.mask_file)
        if sentinel_input is None:
            raise FileNotFoundError("Mask dataset \"%s\" cannot be opened" % args.mask_file)

    for se in raster_scope.extent.subextents(*args.operation_tile_sizes):

        input_data = sentinel_data_utils.load_sentinel_from_dataset(sentinel_input, se, border=0 if args.disable_gaussian_filtering else 3)

        mask_tile: typing.Optional[np.ndarray] = None
        if mask_file is not None:
            scope = input_data.scope
            mask_tile = gdal_array.BandReadAsArray(mask_file.GetRasterBand(1),
                                                   scope.x_off, scope.y_off,
                                                   scope.x_size, scope.y_size,
                                                   scope.x_size, scope.y_size)
            mask_tile = mask_tile == 0

        input_data = sentinel_data_utils.prepare_input_data(input_data, model.required_input_data,
                                                            perform_gauss_filtering=not args.disable_gaussian_filtering,
                                                            aux_mask=mask_tile)

        if ds.GetMetadataItem("DATE") is None:
            ds.SetMetadataItem("DATE", input_data.date)

        flat = {k: v.flatten() for k, v in input_data.data.items()}
        flat['mask'] = input_data.mask.flatten()

        # Dla zachowania pamięci
        input_data.data = None
        input_data.mask = None

        dataframe = pd.DataFrame.from_dict(flat)

        # Dla zachowania pamięci
        del flat

        dataframe = dataframe[dataframe['mask'] > 0]

        # Rearrange
        dataframe = dataframe[model.required_input_data]

        if model.requires_input_data_normalization:
            dataframe = reference_data_utils.normalize_dataset(dataframe, model.input_data_stats)

        result = model.predict(dataframe)

        del dataframe

        result = result[result.isna() == False]

        if len(result) == 0:
            continue

        result[result < 0] = 0

        result[result > args.bathymetry_cutoff] = float('inf')

        x = result.index % input_data.size[0]
        y = np.floor_divide(result.index, input_data.size[0])

        result_arr = np.ndarray(shape=(input_data.size[1], input_data.size[0]), dtype=float)
        result_arr[:, :] = float('nan')

        result_arr[(y, x)] = result

        tile_scope = input_data.scope
        write_scope = se.to_scope(res_geotransform, res_size)

        result_arr = result_arr[tile_scope.y_min_border : (tile_scope.y_min_border + write_scope.y_size), \
                                tile_scope.x_min_border : (tile_scope.x_min_border + write_scope.x_size)]

        gdal_array.BandWriteArray(band, result_arr, xoff=write_scope.x_off, yoff=write_scope.y_off)

    ds.SetGeoTransform(res_geotransform)
    ds.SetSpatialRef(raster_scope.srs)

    del sentinel_input
    del ds
    del mask_file


if __name__ == '__main__':
    main()
