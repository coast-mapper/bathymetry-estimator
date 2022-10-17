import argparse
import itertools

import models

# noinspection PyUnresolvedReferences
import matplotlib_setup
# noinspection PyUnresolvedReferences
import osgeo_setup

import argparse_def
import models_utils
from reference_data_utils import *
from sentinel_data_utils import load_sentinel, prepare_input_data
from models.abstract import input_data, ReportOptions
from scope_utils import Extent


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse_def.define_multiple_input_raster_options(parser)
    argparse_def.define_input_reference_data_options(parser)
    argparse_def.define_calibration_specific_options(parser)
    argparse_def.define_model_specific_options(parser)
    argparse_def.define_plotting_options(parser)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model_cls = models_utils.get_model_class(args.model)
    print("Using model: %s" % args.model)

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

    required_input_data = models_utils.extract_required_input_data_list(model_cls, args)

    samples_coords = pd.concat(map(lambda d: d.data[[input_data.x, input_data.y]], sample_reference_data.values()),
                               ignore_index=True)

    data = map(lambda file: load_sentinel(file,
                                          Extent(min(samples_coords[input_data.x]),min(samples_coords[input_data.y]),
                                                 max(samples_coords[input_data.x]),max(samples_coords[input_data.y]))),
               args.sentinel_data)

    data = list(map(lambda d: prepare_input_data(d, required_input_data, perform_gauss_filtering=not args.disable_gaussian_filtering),
                    data))

    samples = pd.concat(map(lambda t: sample_raster(t[0], t[1]), itertools.product(data, sample_reference_data.values())), ignore_index=True)
    del samples_coords

    validation_samples=None
    if args.validation_data_split:
        if model_cls.model_class_meta_data.supports_validation_data:
            validation_samples = samples.sample(frac=args.validation_data_split[0], random_state=args.validation_data_split[1])
            samples.drop(index=validation_samples.index)

    if args.test_data_split is not None:
        test_samples = samples.sample(frac=args.test_data_split[0], random_state=args.test_data_split[1])
        samples.drop(index=test_samples.index)

    model_options = models_utils.extract_model_args(model_cls, args)

    model: models.abstract.AbstractModel = model_cls(**model_options)

    if hasattr(model, 'srs'):
        srs = next(iter(sample_reference_data.values())).srs
        try:
            setattr(model, 'srs', srs)
        except AttributeError:
            # Read only :(
            pass

    if model_cls.model_class_meta_data.required_input_data_has_ellipsis:
        model.required_input_data = required_input_data

    report_options: ReportOptions = None

    if args.report_dir:
        date = None
        if len(data) == 1:
            date = data[0].date
        report_options = ReportOptions(directory=args.report_dir, label="Train", date=date)

    samples = samples.sample(frac=1)

    if args.normalize_input_data:
        stats = samples.describe()
        stats.pop('depth')
        stats = stats[required_input_data]
        stats = stats.transpose()
        model.input_data_stats = stats


    depth = samples.pop('depth')
    samples = samples[required_input_data]
    if args.normalize_input_data:
        samples = normalize_dataset(samples,stats)
    validation_data = None

    if validation_samples is not None:
        validation_depth = validation_samples.pop('depth')
        validation_samples = validation_samples[required_input_data]
        if args.normalize_input_data:
            validation_samples = normalize_dataset(validation_samples,stats)
        validation_data = (validation_samples,validation_depth)

    fit_res = model.fit(x=samples, y=depth, validation_data=validation_data, report_options=report_options)

    print("Fit results: %s " % fit_res)

    if args.test_data is not None or args.test_data_split is not None:
        if args.test_data is not None:
            test_data = load_reference_data(args.test_data)
            test_data = filter_reference_data(test_data,
                                              minimal=args.reference_data_bounds[0],
                                              maximal=args.reference_data_bounds[1])
            test_samples = pd.concat(map(lambda t: sample_raster(t[0], t[1]), itertools.product(data, [test_data])), ignore_index=True)

        if args.report_dir:
            report_options.label = "Test"

        test_depth = test_samples.pop('depth')
        test_samples = test_samples[required_input_data]
        if args.normalize_input_data:
            test_samples = normalize_dataset(test_samples,stats)
        test_res = model.evaluate(test_samples, test_depth, report_options=report_options)

        print("Test results: %s" % test_res)

    model.save_model(args.model_dir)


if __name__ == "__main__":
    main()
