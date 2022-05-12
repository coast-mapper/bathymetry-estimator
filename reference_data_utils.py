import os
import typing
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from osgeo import osr

from bathymetry_estimations_utils import BathymetryEstimations
from georeferenced_data import GeoreferencedData, _T
from geotransformations import geo_to_pix
from models.abstract import input_data
from sentinel_data_utils import ModelInputData


@dataclass(init=False)
class ReferenceData(GeoreferencedData[pd.DataFrame]):
    """
    source_file_path: indicates from where did we acquire the data
    crs: the coordinate system we operate in
    """
    source_file_path: str
    date: Optional[str] = None

    def __init__(self, data: _T, srs: typing.Union[str, osr.SpatialReference], source_file_path: str,
                 date: Optional[str] = None) -> None:
        super().__init__(data, srs)
        self.source_file_path = source_file_path
        self.date = date


def load_reference_data(data_path: str, delimeter='\t', decimal_char='.') -> ReferenceData:
    """
    Loads reference data from a file. The file has to have supported format.
    Will identify file type by its extension (supported: csv).
    :param data_path: path to data file
    :param delimeter: data delimeter for loading csv
    :param decimal_char: character for decimals
    :return: ReferenceData object
    """
    srs = osr.SpatialReference()
    if os.path.isfile(data_path):
        extension = os.path.split(data_path)[1].split(".")[-1]
        if extension == "csv":
            srs.ImportFromEPSG(32634)
            data_frame = pd.read_csv(data_path, delimiter=delimeter, decimal=decimal_char, header=None)
            data_frame.columns = ["x", "y", "depth"]
            return ReferenceData(data_frame, source_file_path=data_path, srs=srs)

        if extension == "txt":
            srs.ImportFromEPSG(2180)  # from metadata: crs:EPSG 2180
            with open(data_path) as raw_data:
                profile_name = raw_data.readline()
                profile_id = raw_data.readline()
                extract_date = raw_data.readline()
                meta_data = raw_data.readline().split(" ")

                data_from_file = []
                data_line = raw_data.readline().split("\t")
                while len(data_line) > 1:
                    data_from_file.append([float(data_line[3]), float(data_line[2]), float(data_line[1])])
                    data_line = raw_data.readline().split("\t")

                data_frame = pd.DataFrame(data=data_from_file, columns=["x", "y", "depth"])

            return ReferenceData(data_frame, source_file_path=data_path, srs=srs, date=extract_date)

    return ReferenceData(pd.DataFrame(data=[], columns=["x", "y", "depth"]), srs=srs, source_file_path=data_path)


def load_reference_data_folder(data_path: str, delimeter='\t', decimal_char='.') -> Dict[str, ReferenceData]:
    """
    Loads reference data stored in a specified folder
    :param data_path: path to the folder from which we load data. we assume that the folder contains ONLY files that we can read
    :param delimeter: the delimeter for CSV files
    :param decimal_char: the character that splits decimals from fractions
    :return: dictiorary [file_name] -> ReferenceData ; consider that file_name does not contain full path
    """
    if not os.path.isdir(data_path):
        print("err: path does not point to folder: " + data_path)
        return dict()

    return_dict = dict()
    folder_content = os.listdir(data_path)
    for item in folder_content:
        item_path = os.path.join(data_path, item)
        if os.path.isfile(item_path):
            read_data = load_reference_data(item_path, delimeter=delimeter, decimal_char=decimal_char)
            return_dict[item] = read_data

    return return_dict


def filter_reference_data(reference_data: ReferenceData, minimal: float, maximal: float) -> ReferenceData:
    reference_data.data = reference_data.data[
        (reference_data.data['depth'] >= minimal) & (reference_data.data['depth'] < maximal)]
    return reference_data


def sample_raster(data: typing.Union[ModelInputData, BathymetryEstimations],
                  reference_data: ReferenceData) -> pd.DataFrame:
    x_pix, y_pix = geo_to_pix(reference_data.data[input_data.x], reference_data.data[input_data.y], data.geotransform)

    x_pix: pd.Series = np.round(x_pix).astype(int)
    y_pix: pd.Series = np.round(y_pix).astype(int)

    pix_coords: pd.DataFrame = pd.concat([x_pix, y_pix], axis=1, keys=[input_data.x, input_data.y])
    pix_coords = pix_coords[(pix_coords[input_data.x] >= 0) & (pix_coords[input_data.x] < data.size[0]) & (
            pix_coords[input_data.y] >= 0) & (pix_coords[input_data.y] < data.size[1])]

    samples = reference_data.data.loc[pix_coords.index].copy()

    if isinstance(data.data, dict):
        for ds_name, v in data.data.items():
            samples[ds_name] = v[(pix_coords[input_data.y], pix_coords[input_data.x])]

        if input_data.x in data.data.keys():
            samples[input_data.x] = reference_data.data[input_data.x]

        if input_data.y in data.data.keys():
            samples[input_data.y] = reference_data.data[input_data.y]

        for ds_name in data.data.keys():
            samples = samples[samples[ds_name].isna() == False]
    elif isinstance(data, BathymetryEstimations):
        samples['estimation'] = data.data[(pix_coords[input_data.y], pix_coords[input_data.x])]
        samples = samples[samples['estimation'].isna() == False]

    return samples


def normalize_dataset(x: pd.DataFrame, train_stats: pd.DataFrame):
    the_std = np.where(train_stats['std'] == 0, 1, train_stats['std'])
    return (x - train_stats['mean']) / the_std
