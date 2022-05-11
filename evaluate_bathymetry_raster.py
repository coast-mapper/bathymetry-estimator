import argparse
import os
from typing import Dict
import pandas as pd
import numpy as np

import argparse_def
from bathymetry_estimations_utils import load_bathymetry
from models.abstract import input_data
from models.sklearn.metrics import rmse
from reference_data_utils import ReferenceData, load_reference_data, filter_reference_data, load_reference_data_folder, \
    sample_raster
from scope_utils import Extent
from matplotlib_setup import plt
from plots import fit_scatter


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparse_def.define_input_bathymetry_raster_options(parser)
    argparse_def.define_input_reference_data_options(parser)
    argparse_def.define_bathymetry_raster_evaluate_specific_options(parser)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

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

    samples_coords = pd.concat(map(lambda d: d.data[[input_data.x, input_data.y]], sample_reference_data.values()),
                               ignore_index=True)

    data = load_bathymetry(args.bathymetry_estimation,
                         Extent(min(samples_coords[input_data.x]), min(samples_coords[input_data.y]),
                                max(samples_coords[input_data.x]), max(samples_coords[input_data.y])))
    del samples_coords

    samples = pd.concat(map(lambda d: sample_raster(data, d), sample_reference_data.values()), ignore_index=True)

    samples['diff'] = samples['depth'] - samples['estimation']
    samples['abs_diff'] = np.abs(samples['diff'])

    samples.sort_values('abs_diff', inplace=True)

    best = samples.head(int(np.ceil(args.best_predictions_fraction * len(samples))))

    rmse_res = rmse(best['depth'], best['estimation'])

    print(f"RMSE: {rmse_res}")

    if args.report_dir:
        os.makedirs(args.report_dir, exist_ok=True)
        with open(os.path.join(args.report_dir,'report.txt'),'w') as f:
            print(f"RMSE: {rmse_res}", file=f)
            print(f"Data points: {len(samples)}", file=f)

        fit_scatter(best['depth'], best['estimation'], 'bathymetry [m]','estimation [m]',title=data.date)
        plt.savefig(os.path.join(args.report_dir,'scatter.png'))
        plt.savefig(os.path.join(args.report_dir,'scatter.svg'))


if __name__ == '__main__':
    main()