import abc
import argparse
import dataclasses
import typing
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from osgeo import ogr
from osgeo import osr

import argparse_types
import models.abstract
import scope_utils
from argparse_ext import DeprecatedStoreAction
from plots import fit_scatter
from . import _metrics
from . import _weights
from ._inverse import calc_inv


def wkt_model_centers_with_radius(wkt: str) -> typing.List[typing.Tuple[float, float, float]]:
    geom: ogr.Geometry = ogr.CreateGeometryFromWkt(wkt)

    if geom is None:
        raise argparse.ArgumentTypeError("\"%s\" could not be parsed as WKT text." % wkt)

    gtype = geom.GetGeometryType()

    if gtype != ogr.wkbMultiPointM:
        raise argparse.ArgumentTypeError("\"%s\" should be a WKT multipoint with M object." % wkt)

    ret = []
    for i in range(geom.GetGeometryCount()):
        point: ogr.Geometry = geom.GetGeometryRef(i)
        ret.append((point.GetX(), point.GetY(), point.GetM()))

    return ret


class RegressionType(abc.ABC):
    linear = 'linear'
    exponential = 'exponential'


class GeographyWeightedModelMeta(models.abstract.AbstractModelMeta):

    def define_parameters(cls, parser: argparse.ArgumentParser):
        if cls == GeographyWeightedModel:
            group = parser.add_argument_group("Geography Weighted Model")
            group.add_argument("--gwm-mode", choices=[RegressionType.linear, RegressionType.exponential],
                               default=RegressionType.linear,
                               help="Type of regression used for local models. For linear model is in form a*x+b, for exponential is exp(a*x+b).")
            group.add_argument("--gwm-models-centers", type=wkt_model_centers_with_radius, required=False,
                               help="Model centers in WKT format using MULTIPOINT type. Example: MULTIPOINT M((10 40 300), (40 30 300), (20 20 300), (30 10 400))")
            group.add_argument("--gwm-srs", type=argparse_types.coordinate_reference_system, required=False,
                               help="Model centers coordinates system in WKT format.")
            group.add_argument("--gwm-models-centers-source", choices=['compute', 'predefined'], default='predefined',
                               help="How models centers are determined. Compute is deprecated")
            group.add_argument("--gwm-no-of-local-models", type=int, action=DeprecatedStoreAction,
                               help="(Deprecated) Number of local models. Used in case 'compute' option for models center source")
            group.add_argument("--gwm-local-model-range", type=int, action=DeprecatedStoreAction,
                               help="(Deprecated) Range of local models in meters.")

    @property
    def model_class_meta_data(cls) -> models.abstract.ModelClassMetaData:
        return models.abstract.ModelClassMetaData(required_input_data=[models.abstract.input_data.x,
                                                                       models.abstract.input_data.y,
                                                                       models.abstract.input_data.raw_bathymetry],
                                                  parameter_prefixes=['gwm_'])

    def _load_model(cls, path: str, configuration: typing.Dict[str, typing.Any]) -> models.abstract.AbstractModel:
        import os

        data: pd.DataFrame = pd.read_csv(os.path.join(path, 'model.csv'))

        if 'r' not in data.columns:

            if 'gwm_local_model_range' not in configuration:
                raise RuntimeError("There is no radius (r) column in model.csv")
            else:
                fallback_radius = configuration['gwm_local_model_range']
                warnings.warn(
                    'Radius column not defined in model.csv using deprecated gwm_local_model_range as fallback.'
                )
                data['r'] = fallback_radius

        lms = [LocalModel(center=(row['x'], row['y']), radius=row['r'], a=row['a'], b=row['b']) for inx, row in
               data.iterrows()]

        # Drop suspicious keys
        suspicious_keys = ['gwm_models_centers',
                           '_local_models',
                           'gwm_no_of_local_models',
                           'gwm_models_centers_source',
                           'gwm_local_model_range']

        for key in suspicious_keys:
            if key in configuration:
                del configuration[key]

        configuration['_local_models'] = lms

        if 'gwm_srs' in configuration:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(configuration['gwm_srs'])
            configuration['gwm_srs'] = srs

        return cls(**configuration)


@dataclasses.dataclass
class LocalModel(object):
    center: typing.Tuple[float, float]
    radius: float
    a: float = 0.0
    b: float = 0.0


class GeographyWeightedModel(models.abstract.AbstractModel, metaclass=GeographyWeightedModelMeta):

    def __init__(self,
                 gwm_models_centers_source: typing.Optional[str] = None,
                 gwm_models_centers: typing.Optional[typing.List[typing.Tuple[float, float, float]]] = None,
                 gwm_no_of_local_models: typing.Optional[int] = None,
                 gwm_local_model_range: typing.Optional[int] = None,
                 gwm_srs: typing.Optional[osr.SpatialReference] = None,
                 gwm_mode: str = RegressionType.linear,
                 _local_models: typing.Optional[typing.List[LocalModel]] = None,
                 *args, **kwargs):
        if _local_models is None:
            if gwm_local_model_range is not None:
                warnings.warn("Global range for GWM models is deprecated.", DeprecationWarning, stacklevel=2)
            self._local_model_range = gwm_local_model_range

            if gwm_models_centers_source == 'compute':
                warnings.warn("Computing local centers in GWM model is deprecated.", DeprecationWarning, stacklevel=2)
                self._need_to_compute_centers = True
                self._no_local_models = gwm_no_of_local_models
                self._local_model_range = gwm_local_model_range
            elif gwm_models_centers_source == 'predefined':
                self._need_to_compute_centers = False
                if gwm_models_centers is not None:
                    self._local_models = [LocalModel(center=(x, y), radius=r) for x, y, r in gwm_models_centers]
                    self._no_local_models = len(self._local_models)
                else:
                    raise RuntimeError("Model centers not defined")
        else:
            self._local_models = _local_models
            self._no_local_models = len(self._local_models)

        self._regression_mode = gwm_mode
        self._srs = gwm_srs
        super().__init__(*args, **kwargs)

    def __calc_metrics(self, x: pd.Series, y: pd.Series):
        metrics = [_metrics.mean_squared_error, _metrics.mean_percent_error, _metrics.r2_score]
        ret = {}

        for m in metrics:
            ret[m.__name__] = m(x, y)

        return ret

    @property
    def local_models(self) -> typing.List[LocalModel]:
        return self._local_models

    @property
    def srs(self) -> typing.Optional[osr.SpatialReference]:
        return self._srs

    @srs.setter
    def srs(self, value: osr.SpatialReference):
        self._srs = value

    @property
    def model_extent(self) -> typing.Optional[scope_utils.Extent]:
        x_ = []
        y_ = []

        for lm in self._local_models:
            x_.append(lm.center[0] - lm.radius)
            x_.append(lm.center[0] + lm.radius)

            y_.append(lm.center[1] - lm.radius)
            y_.append(lm.center[1] + lm.radius)

        return scope_utils.Extent(min(x_), min(y_), max(x_), max(y_))

    def get_config(self) -> typing.Dict[str, typing.Any]:
        conf = {
            **super().get_config(),
            'gwm_mode': self._regression_mode
        }

        if self.srs is not None:
            conf['gwm_srs'] = self.srs.ExportToWkt()

        return conf

    def save_model(self, path: str):
        super().save_model(path)
        import os

        with open(os.path.join(path, 'model.csv'), 'w') as f:
            print("x,y,r,a,b", file=f)
            for lm in self.local_models:
                print("%f,%f,%f,%f,%f" % (lm.center[0], lm.center[1], lm.radius, lm.a, lm.b), file=f)

    def fit(self, x: pd.DataFrame, y: pd.Series,
            validation_data: typing.Optional[typing.Tuple[pd.DataFrame, pd.Series]] = None,
            report_options: typing.Optional[models.abstract.ReportOptions] = None) -> models.abstract.FitResults:
        if self._need_to_compute_centers:
            print("Calculating local centers")
            from ._centers import calculate_local_centers
            centers = calculate_local_centers(x[[models.abstract.input_data.x, models.abstract.input_data.y]],
                                              self._no_local_models)
            self._local_models = [LocalModel((row['x'], row['y']), self._local_model_range) for k, row in centers.iterrows()]

        coord_x = x[models.abstract.input_data.x]
        coord_y = x[models.abstract.input_data.y]
        raw_bathymery = x[models.abstract.input_data.raw_bathymetry]

        if self._regression_mode == RegressionType.linear:
            y_ = y.to_numpy()
        else:
            y_ = np.log(y.to_numpy())

        for lm in self._local_models:
            weights = _weights.calc_2D_weights(coord_x, coord_y, lm.center, lm.radius)

            res = calc_inv(raw_bathymery.to_numpy(), y_, weights.to_numpy(), )
            lm.a = res[1]
            lm.b = res[0]

        y_pred = self.predict(x)

        if report_options is not None:
            _evaluation_report(report_options.directory, y_pred, y, report_options.label, report_options.date)

        return models.abstract.FitResults(metrics_values=self.__calc_metrics(y, y_pred))

    def evaluate(self, x: pd.DataFrame, y: pd.Series,
                 report_options: typing.Optional[
                     models.abstract.ReportOptions] = None) -> models.abstract.EvaluateResults:
        y_pred = self.predict(x)
        if report_options is not None:
            _evaluation_report(report_options.directory, y_pred, y, report_options.label, report_options.date)
        return models.abstract.EvaluateResults(self.__calc_metrics(y, y_pred))

    def predict(self, x: pd.DataFrame) -> pd.Series:
        coord_x = x[models.abstract.input_data.x]
        coord_y = x[models.abstract.input_data.y]
        raw_bathymery = x[models.abstract.input_data.raw_bathymetry]
        weights_ = pd.Series(index=x.index, dtype=float)
        weights_[:] = 0.0
        results_ = pd.Series(index=x.index, dtype=float)
        results_[:] = 0.0

        for lm in self._local_models:
            w_ = _weights.calc_2D_weights(coord_x, coord_y, lm.center, lm.radius)
            r_ = (raw_bathymery * lm.a + lm.b) * w_

            weights_ += w_
            results_ += r_

        results_ /= weights_
        if self._regression_mode == RegressionType.exponential:
            results_ = np.exp(results_)
        return results_

    def __str__(self):
        return "%s(local_models=%s, srs=%s)" % (self.__class__.__name__, self.local_models, self.srs)

    __repr__ = __str__


def _evaluation_report(repors_dir: str, y_pred: pd.Series, y: pd.Series, label: str, date: typing.Optional[str]):
    import os

    os.makedirs(repors_dir, exist_ok=True)

    report = open(os.path.join(repors_dir, '%s_report.txt' % label), "w")

    print("====Evaluation report for %s data====" % label, file=report)
    print("Correlation: %s" % np.corrcoef(y_pred, y), file=report)

    rmse = _metrics.mean_squared_error(y, y_pred)
    print("RMSE: %s" % rmse, file=report)

    R_2 = _metrics.r2_score(y, y_pred)
    print("R^2: %s" % R_2, file=report)

    mpe = _metrics.mean_percent_error(y, y_pred)

    print("MPE: %s" % mpe, file=report)

    fig = fit_scatter(y, y_pred, "%s data [m]" % label, "%s prediction [m]" % label, date)
    plt.savefig(os.path.join(repors_dir, "%s_scatter.png" % label))
    plt.savefig(os.path.join(repors_dir, "%s_scatter.svg" % label))
    plt.close(fig)
    fig = fit_scatter(y, y_pred, "%s data [m]" % label, "%s prediction [m]" % label, date, without_description=True)
    plt.savefig(os.path.join(repors_dir, "%s_scatter_raw.png" % label))
    plt.savefig(os.path.join(repors_dir, "%s_scatter_raw.svg" % label))
    plt.close(fig)
