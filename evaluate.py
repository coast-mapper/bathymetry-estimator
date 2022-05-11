import argparse
from typing import Dict
import pandas as pd

# noinspection PyUnresolvedReferences
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
    argparse_def.define_evaluate_specific_options(parser)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model = models.abstract.AbstractModel.load_model(args.model_dir)

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

    report_opt: models.abstract.ReportOptions = None

    if args.report_dir:
        report_opt = models.abstract.ReportOptions(directory=args.report_dir, label="Evaluate", date=data.date)

    depth = samples.pop('depth')
    samples = samples[required_input_data]

    if model.requires_input_data_normalization:
        samples = reference_data_utils.normalize_dataset(samples, model.input_data_stats)

    res = model.evaluate(samples, depth, report_options=report_opt)

    print("Evaluate result: %s" % res)


if __name__ == '__main__':
    main()
