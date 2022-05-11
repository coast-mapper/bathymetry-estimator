import argparse
import itertools
from typing import Dict
import pandas as pd

# noinspection PyUnresolvedReferences
import typing

import matplotlib_setup
# noinspection PyUnresolvedReferences
import osgeo_setup

import argparse_def
import reference_data_utils
from models.abstract import input_data
from scope_utils import Extent
from reference_data_utils import ReferenceData, load_reference_data, filter_reference_data, load_reference_data_folder, \
    sample_raster
import models.abstract
from sentinel_data_utils import load_sentinel, prepare_input_data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse_def.define_input_raster_options(parser)
    argparse_def.define_input_reference_data_options(parser)
    argparse_def.define_data_split_options(parser)
    argparse_def.define_evaluate_specific_options(parser)
    argparse_def.define_ploting_options(parser)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model = models.abstract.AbstractModel.load_model(args.model_dir)
    model_cls = model.__class__

    sample_reference_data: Dict[str, ReferenceData] = None
    if args.reference_data:
        sample_reference_data_ = load_reference_data(args.reference_data)
        sample_reference_data = {args.reference_data: filter_reference_data(sample_reference_data_,
                                                                            minimal=args.reference_data_bounds[0],
                                                                            maximal=args.reference_data_bounds[1])}

    if args.reference_data_folder:
        sample_reference_data = load_reference_data_folder(args.reference_data_folder)
        for ref_dat in sample_reference_data:
            sample_reference_data[ref_dat] = filter_reference_data(sample_reference_data[ref_dat],
                                                                   minimal=args.reference_data_bounds[0],
                                                                   maximal=args.reference_data_bounds[1])

    if args.reference_data_srs is not None:
        for v in sample_reference_data.values():
            v.srs = args.reference_data_srs

    required_input_data = model.required_input_data

    samples_coords = pd.concat(map(lambda d: d.data[[input_data.x, input_data.y]], sample_reference_data.values()),
                               ignore_index=True)

    data = load_sentinel(args.sentinel_data,
                         Extent(min(samples_coords[input_data.x]), min(samples_coords[input_data.y]),
                                max(samples_coords[input_data.x]), max(samples_coords[input_data.y])))
    del samples_coords
    data = prepare_input_data(data, required_input_data, perform_gauss_filtering=True)

    samples = pd.concat(map(lambda d: sample_raster(data, d), sample_reference_data.values()), ignore_index=True)

    validation_samples=None
    if args.validation_data_split:
        if model_cls.model_class_meta_data.supports_validation_data:
            validation_samples = samples.sample(frac=args.validation_data_split[0], random_state=args.validation_data_split[1])
            samples.drop(index=validation_samples.index)

    test_samples=None
    if args.test_data_split is not None:
        test_samples = samples.sample(frac=args.test_data_split[0], random_state=args.test_data_split[1])
        samples.drop(index=test_samples.index)
    elif args.test_data is not None:
        test_data = load_reference_data(args.test_data)
        test_data = filter_reference_data(test_data,
                                          minimal=args.reference_data_bounds[0],
                                          maximal=args.reference_data_bounds[1])
        test_samples = pd.concat(map(lambda t: sample_raster(t[0], t[1]), itertools.product(data, [test_data])), ignore_index=True)

    data_dict = {}

    data_dict["Train"] = samples
    if validation_samples is not None:
        data_dict["Validation"] = validation_samples
    if test_samples is not None:
        data_dict["Test"] = test_samples

    for label, s in data_dict.items():
        report_opt: typing.Optional[models.abstract.ReportOptions] = None

        if args.report_dir:
            report_opt = models.abstract.ReportOptions(directory=args.report_dir, label=label, date=data.date)

        depth = s.pop('depth')
        s = s[required_input_data]

        if model.requires_input_data_normalization:
            s = reference_data_utils.normalize_dataset(s, model.input_data_stats)

        res = model.evaluate(s, depth, report_options=report_opt)

        print(f"{label} result: {res}")


if __name__ == '__main__':
    main()
