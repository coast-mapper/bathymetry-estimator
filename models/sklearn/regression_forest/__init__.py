import argparse
import typing

import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble

import models.sklearn
import models.sklearn.metrics
from models.abstract import ModelClassMetaData, ReportOptions, EvaluateResults, FitResults


class RegressionForestModelMeta(models.sklearn.SklearnModelMeta):

    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        as_dict = {**super().model_class_meta_data.__dict__}
        as_dict['parameter_prefixes'].append('srf_')

        return ModelClassMetaData(**as_dict)

    def define_parameters(cls, parser: argparse.ArgumentParser):
        if cls == RegressionForestModel:
            group = parser.add_argument_group("Regression Forest Model")
            group.add_argument("--srf-number-of-estimators", type=int, default=100)
            group.add_argument("--srf-max-depth", type=int, default=100)
            group.add_argument("--srf-min-samples-split", type=float, default=0.01)
            group.add_argument("--srf-min-samples-leaf", type=float, default=0.001)


class RegressionForestModel(models.sklearn.SklearnModel[sklearn.ensemble.RandomForestRegressor],
                            metaclass=RegressionForestModelMeta):
    def __init__(self,
                 srf_number_of_estimators: float = 100,
                 srf_max_depth: float = 100,
                 srf_min_samples_split: float = 0.01,
                 srf_min_samples_leaf: float = 0.001,
                 _model: typing.Optional[sklearn.ensemble.RandomForestRegressor] = None,
                 *args, **kwargs):
        if _model is None:
            _model = sklearn.ensemble.RandomForestRegressor(
                n_estimators=srf_number_of_estimators,
                max_depth=srf_max_depth,
                min_samples_split=srf_min_samples_split,
                min_samples_leaf=srf_min_samples_leaf
            )
        super().__init__(_model,*args, **kwargs)

    def _fit_on_model(self, model: sklearn.tree.DecisionTreeRegressor, x: pd.DataFrame, y: pd.Series,
                      report_options: typing.Optional[ReportOptions] = None) -> FitResults:
        model.fit(x, y)
        y_pred = self._predict_on_model(model, x)
        metric_values = {m: getattr(models.sklearn.metrics, m)(y, y_pred) for m in models.sklearn.metrics.__all__}
        res = FitResults(metrics_values=metric_values)

        if report_options is not None:
            self._create_report(model, x, y, y_pred, report_options, res)
        return res

    def _evaluate_on_model(self, model: sklearn.tree.DecisionTreeRegressor, x: pd.DataFrame, y: pd.Series,
                           report_options: typing.Optional[ReportOptions] = None) -> EvaluateResults:
        y_pred = self._predict_on_model(model, x)
        metric_values = {m: getattr(models.sklearn.metrics, m)(y, y_pred) for m in models.sklearn.metrics.__all__}
        res = EvaluateResults(metrics_values=metric_values)

        if report_options is not None:
            self._create_report(model, x, y, y_pred, report_options, res)

        return res

    def _predict_on_model(self, model: sklearn.tree.DecisionTreeRegressor, x: pd.DataFrame) -> pd.Series:
        y_pred = model.predict(x)
        y_pred = np.array(y_pred)
        y_pred = np.reshape(y_pred, len(y_pred))
        y_pred = pd.Series(y_pred, index=x.index)
        return y_pred
