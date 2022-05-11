import abc
import argparse
import typing

import numpy as np
import pandas as pd
from osgeo import osr

import argparse_types
import models.abstract
from models.abstract import ModelClassMetaData, ReportOptions, FitResults, EvaluateResults
from models.geography_weighted_model import wkt_model_centers_with_radius
from scope_utils import Extent

# Import bez importu :)
if False:
    import tensorflow.keras as keras


class GeographyWeightedKerasModelMeta(models.keras.KerasModelMeta):
    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        base = super().model_class_meta_data.__dict__
        base['required_input_data'] = [models.abstract.input_data.x,
                                       models.abstract.input_data.y,
                                       ...
                                       ]
        base['parameter_prefixes'] = ['gwmk_', *base['parameter_prefixes']]

        return ModelClassMetaData(**base)

    def define_parameters(cls, parser: argparse.ArgumentParser):
        if cls == GeographyWeightedKerasModel:
            group = parser.add_argument_group("Geography Weighted Model (Keras implementation)")
            # group.add_argument("--gwmk-mode", choices=[RegressionType.linear, RegressionType.exponential],
            #                    default=RegressionType.linear,
            #                    help="Type of regression used for local models. For linear model is in form a*x+b, for exponential is exp(a*x+b).")
            group.add_argument("--gwmk-models-centers", type=wkt_model_centers_with_radius, required=False,
                               help="Model centers in WKT format using MULTIPOINT type. Example: MULTIPOINT M((10 40 300), (40 30 300), (20 20 300), (30 10 400))")
            group.add_argument("--gwmk-srs", type=argparse_types.coordinate_reference_system, required=False,
                               help="Model centers coordinates system in WKT format.")
            group.add_argument("--gwmk-train-mode", type=str, required=False, choices=['separate', 'simultaneous'],
                               default='simultaneous')

    @property
    def custom_objects(cls) -> typing.Dict[str, typing.Any]:
        from .impl import WeightLayer, GWMKeras
        from models.keras.linear_model.layer import LinearLayer
        objs = super().custom_objects
        objs['WeightLayer'] = WeightLayer
        objs['LinearLayer'] = LinearLayer
        objs['GWMKeras'] = GWMKeras
        return objs

    def _refine_configuration(cls, configuration: typing.Dict[str, typing.Any]):
        configuration = super()._refine_configuration(configuration)
        keys = ['gwmk_mode', 'gwmk_models_centers']

        for k in keys:
            if k in configuration:
                del configuration[k]

        if 'gwmk_srs' in configuration:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(configuration['gwmk_srs'])
            configuration['gwmk_srs'] = srs

        if 'gwmk_train_mode' not in configuration:
            configuration['gwmk_train_mode'] = 'simultaneous'

        return configuration


class GeographyWeightedKerasModel(models.keras.KerasModel, metaclass=GeographyWeightedKerasModelMeta):

    def __init__(self,
                 gwmk_train_mode: str,
                 gwmk_srs: osr.SpatialReference,
                 gwmk_models_centers: typing.Optional[typing.List[typing.Tuple[float, float, float]]] = None,
                 *args, **kwargs):
        self._disable_filter_in_predict = False
        self._srs = gwmk_srs
        self._train_mode = gwmk_train_mode
        if gwmk_models_centers is not None:
            kwargs['_model'] = self._create_model(gwmk_models_centers)
        super().__init__(*args, **kwargs)

    def get_config(self) -> typing.Dict[str, typing.Any]:
        conf = super().get_config()

        if self.srs is not None:
            conf['gwmk_srs'] = self.srs.ExportToWkt()
        conf['gwmk_train_mode'] = self._train_mode

        return conf

    @property
    def srs(self) -> typing.Optional[osr.SpatialReference]:
        return self._srs

    @srs.setter
    def srs(self, value: osr.SpatialReference):
        self._srs = value

    @property
    def _model_centers(self):
        from .impl import WeightLayer
        return map(lambda wl: (wl.x, wl.y, wl.r), filter(lambda l: isinstance(l, WeightLayer), self.model.layers))

    @property
    def model_extent(self) -> typing.Optional[Extent]:
        x_ = []
        y_ = []
        for x, y, r in self._model_centers:
            x_.append(x + r)
            x_.append(x - r)

            y_.append(y + r)
            y_.append(y - r)

        return Extent(x_min=min(x_), x_max=max(x_), y_min=min(y_), y_max=max(y_))

    def _filter_data_spatialy(self, input: pd.DataFrame, y_true: typing.Optional[pd.Series]) -> (
            pd.DataFrame, typing.Optional[pd.Series]):
        print("[%s] filtering data spatially." % self.__class__.__name__)
        x_ = input[models.abstract.input_data.x]
        y_ = input[models.abstract.input_data.y]

        sum = pd.Series(False, index=input.index)

        for x, y, r in self._model_centers:
            cond = np.power(x_ - x, 2) + np.power(y_ - y, 2) < np.power(r, 2)
            sum = np.logical_or(sum, cond)

        return input.loc[sum], y_true.loc[sum] if y_true is not None else None

    def _create_model(self, models_centers: typing.List[typing.Tuple[float, float, float]]):

        from .impl import GWMKeras

        return GWMKeras(model_centers=models_centers, model_creator=self._create_submodel)

    @abc.abstractmethod
    def _create_submodel(self) -> 'keras.Model':
        pass

    def __filter_input_by_weights(self, weights, x, y):
        from models.abstract import input_data
        sub_cols = [c for c in x.columns if c != input_data.x and c != input_data.y]
        non_zero = weights > 0
        sub_x = x[sub_cols]
        sub_x = sub_x[non_zero]
        sub_y = y[non_zero]
        we = weights[non_zero]
        return sub_x, sub_y, we

    def __calc_weights(self, w_layer, x) -> pd.Series:
        from models.abstract import input_data
        numpy_res: np.ndarray = w_layer(x[[input_data.x, input_data.y]].to_numpy()).numpy()
        numpy_res = numpy_res.reshape(len(numpy_res))
        return pd.Series(numpy_res, index=x.index)

    def __rename_labels(self, prefix: str, fr: FitResults) -> FitResults:
        renamed_metrics = {prefix + k: v for k, v in fr.metrics_values.items()}
        new_columns_names = {c: prefix + c for c in fr.history.columns}
        renamed_hitory = fr.history.rename(columns=new_columns_names)

        return FitResults(renamed_metrics, renamed_hitory)

    def fit(self, x: pd.DataFrame, y: pd.Series,
            validation_data: typing.Optional[typing.Tuple[pd.DataFrame, pd.Series]] = None,
            report_options: typing.Optional[ReportOptions] = None) -> FitResults:
        self._disable_filter_in_predict = True
        x, y = self._filter_data_spatialy(x, y)
        if validation_data is not None:
            validation_data = self._filter_data_spatialy(*validation_data)

        if self._train_mode == 'simultaneous':
            ret = super().fit(x, y, validation_data, report_options)
        else:
            from .impl import GWMKeras
            from models.abstract import input_data
            import os
            main_model: GWMKeras = self.model

            m_n_w = main_model.models_and_weights

            fit_results: typing.List[FitResults] = []

            for i in range(len(m_n_w)):
                submodel, w_layer = m_n_w[i]
                weights = self.__calc_weights(w_layer, x)
                sub_x, sub_y, weights = self.__filter_input_by_weights(weights, x, y)

                validation_w = None
                sub_validation_data = None

                if validation_data is not None:
                    vx = validation_data[0]
                    validation_w = self.__calc_weights(w_layer, vx)
                    vsub_x, vsub_y, validation_w = self.__filter_input_by_weights(validation_w, *validation_data)
                    sub_validation_data = (vsub_x, vsub_y)

                report_opt = None

                if report_options is not None:
                    d = {**report_options.__dict__}
                    d['directory'] = os.path.join(d['directory'], "model_%d" % (i,))
                    report_opt = ReportOptions(**d)

                print("Fitting model %d" % (i,))
                fr = self._fit_on_model(submodel, sub_x, sub_y, sub_validation_data, report_opt, weights, validation_w)
                fit_results.append(self.__rename_labels('model_%d_' % (i,), fr))

            merged_metrics = {}

            for f in fit_results:
                merged_metrics.update(f.metrics_values)

            merged_history = pd.concat(map(lambda f: f.history, fit_results), axis=1, sort=False)

            ret = FitResults(
                merged_metrics,
                merged_history
            )

            self._check_model(self.model)
            self._create_fit_report(self.model, x, y, validation_data, ret, report_options)

        self._disable_filter_in_predict = False
        return ret

    def evaluate(self, x: pd.DataFrame, y: pd.Series,
                 report_options: typing.Optional[ReportOptions] = None) -> EvaluateResults:
        self._disable_filter_in_predict = True
        x, y = self._filter_data_spatialy(x, y)
        ret = super().evaluate(x, y, report_options)
        self._disable_filter_in_predict = False
        return ret

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if not self._disable_filter_in_predict:
            original_index = x.index
            x, e = self._filter_data_spatialy(x, None)
        ret = super().predict(x)

        if not self._disable_filter_in_predict:
            new_ret = pd.Series(float('nan'), index=original_index)
            new_ret.loc[ret.index] = ret
            return new_ret
        return ret
