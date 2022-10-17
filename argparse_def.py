import argparse
import argparse_types
import models_utils
import models.geography_weighted_model
import models.keras
import models.keras.linear_model
import models.keras.neural_network
import models.keras.geography_weighted_model_linear
import models.keras.geography_weighted_model_neural_network
import models.sklearn
import models.sklearn.regression_tree
import models.sklearn.regression_forest

from argparse_ext import ModelHelp, SetFigureSizeAction


def define_multiple_input_raster_options(parser: argparse.ArgumentParser):
    parser.add_argument("--sentinel-data", required=True, type=str,
                        help="GDAL dataset path for sentinel", action='append')
    parser.add_argument("--disable-gaussian-filtering", action="store_const", const=True)


def define_input_raster_options(parser: argparse.ArgumentParser):
    parser.add_argument("--sentinel-data", required=True, type=str,
                        help="GDAL dataset path for sentinel")
    parser.add_argument("--disable-gaussian-filtering", action="store_const", const=True)


def define_input_bathymetry_raster_options(parser: argparse.ArgumentParser):
    parser.add_argument("--bathymetry-estimation", required=True, type=str,
                        help="GDAL dataset with bathymetry estimation")


def define_input_reference_data_options(parser: argparse.ArgumentParser):
    reference_data_g = parser.add_mutually_exclusive_group(required=True)

    reference_data_g.add_argument("--reference-data", required=False, type=str, help="Path to reference data csv file")
    reference_data_g.add_argument("--reference-data-folder", required=False, type=str,
                                  help="Path to reference data folder with txt data files")
    parser.add_argument("--reference-data-srs", required=False, type=argparse_types.coordinate_reference_system,
                        help="Optionally you can override default reference data srs. Required is WKT, proj "
                             "or EPSG number")
    parser.add_argument("--reference-data-bounds", type=argparse_types.separated_list(dtype=float), default=[2.0, 10.0])


def define_calibration_specific_options(parser: argparse.ArgumentParser):
    from models.abstract import input_data
    input_data_list = [getattr(input_data, a) for a in vars(input_data) if not a.startswith('_')]
    models_with_ellipsis = [mcls.__name__ for mcls in models_utils.get_available_non_abstract_models().values()
                            if mcls.model_class_meta_data.required_input_data is Ellipsis
                            or (isinstance(mcls.model_class_meta_data.required_input_data, list)
                                and Ellipsis in mcls.model_class_meta_data.required_input_data)]

    models_with_ellipsis_text = str(models_with_ellipsis) if len(models_with_ellipsis) > 0 else "None"

    model_names = models_utils.get_available_non_abstract_models().keys()
    parser.add_argument("--model", "-m", type=str, choices=model_names,
                        help="Bathymetry model which will be calibrated",
                        default=models.geography_weighted_model.GeographyWeightedModel.__name__)
    parser.add_argument("--input-data", type=argparse_types.separated_list(allowed_values=input_data_list),
                        help="Comma separated list of input data for trained model. "
                             "Available input data types are: %s. "
                             "Currently following models support free input data setup: %s." % (
                                 input_data_list, models_with_ellipsis_text),
                        default=[input_data.x, input_data.y, input_data.raw_bathymetry])
    parser.add_argument("--report-dir", type=str, help="Path to report directory")
    parser.add_argument("--model-dir", type=str, help="Path where model will be saved")
    parser.add_argument("--model-help", type=str, action=ModelHelp, choices=model_names, required=False,
                        help="Displays information about selected model")

    define_data_split_options(parser)

    parser.add_argument("--normalize-input-data", action="store_const", const=True)


def define_data_split_options(parser: argparse.ArgumentParser):
    parser.add_argument("--validation-data-split", required=False, type=argparse_types.random_sample_configuration,
                        help="With this option validation data would be extracted from train data. You should provide "
                             "fraction and optionally random state in such format: frac[,state] .")
    test_data_g = parser.add_mutually_exclusive_group(required=False)
    test_data_g.add_argument("--test-data", required=False, type=str, help="Path to test data csv file")
    test_data_g.add_argument("--test-data-split", required=False, type=argparse_types.random_sample_configuration,
                             help="With this option test data would be extracted from train data. You should provide "
                                  "fraction and optionally random state in such format: frac[,state] .")


def define_plotting_options(parser: argparse.ArgumentParser):
    parser.add_argument("--figure-size", required=False, type=argparse_types.separated_tuple([float, float], ','),
                        action=SetFigureSizeAction,
                        help="Default size of figure in inches.")


def define_prediction_specific_options(parser: argparse.ArgumentParser):
    parser.add_argument("--model-dir", type=str, required=True, help="Path model, which be used to estimate bathymetry")
    parser.add_argument("--bathymetry-cutoff", type=float,
                        help="Value in meters above which bathymetry is changed to Inf",
                        default=16)
    parser.add_argument("--preserve-original-size", required=False, action="store_const", const=True,
                        help="With this parameter result image will have same size as input image")
    parser.add_argument("--result-file", type=str, required=True, help="Result file location")
    parser.add_argument("--mask-file", type=str, required=False, help="File which contains sea/land mask")
    parser.add_argument("--operation-tile-sizes", type=argparse_types.separated_tuple([int, int]),
                        required=False,
                        default=(5000, 5000),
                        help="Size in raster srs units of single tile processed at once.")


def define_evaluate_specific_options(parser: argparse.ArgumentParser):
    parser.add_argument("--model-dir", type=str, required=True, help="Path model, which be used to estimate bathymetry")
    define_report_options(parser)


def define_bathymetry_raster_evaluate_specific_options(parser: argparse.ArgumentParser):
    parser.add_argument("--best-predictions-fraction", type=float, default=.95)
    define_report_options(parser)


def define_report_options(parser: argparse.ArgumentParser):
    parser.add_argument("--report-dir", type=str, help="Path to report directory")


def define_model_specific_options(parser: argparse.ArgumentParser):
    for model in models_utils.get_available_models().values():
        model.define_parameters(parser)
