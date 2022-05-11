import argparse
import os
import typing

import numpy as np
import pandas as pd
import sklearn
import sklearn.tree

import models.sklearn
import models.sklearn.metrics
from models.abstract import ModelClassMetaData, ReportOptions, EvaluateResults, FitResults


class RegressionTreeModelMeta(models.sklearn.SklearnModelMeta):

    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        as_dict = {**super().model_class_meta_data.__dict__}
        as_dict['parameter_prefixes'].append('srt_')

        return ModelClassMetaData(**as_dict)

    def define_parameters(cls, parser: argparse.ArgumentParser):
        if cls == RegressionTreeModel:
            group = parser.add_argument_group("Regression Tree Model")
            group.add_argument("--srt-max-depth", type=int, default=100)
            group.add_argument("--srt-min-samples-split", type=float, default=0.01)
            group.add_argument("--srt-min-samples-leaf", type=float, default=0.001)


class RegressionTreeModel(models.sklearn.SklearnModel[sklearn.tree.DecisionTreeRegressor],
                          metaclass=RegressionTreeModelMeta):
    def __init__(self,
                 srt_max_depth: float = 100,
                 srt_min_samples_split: float = 0.01,
                 srt_min_samples_leaf: float = 0.001,
                 _model: typing.Optional[sklearn.tree.DecisionTreeRegressor] = None,
                 *args, **kwargs):
        if _model is None:
            _model = sklearn.tree.DecisionTreeRegressor(
                max_depth=srt_max_depth,
                min_samples_split=srt_min_samples_split,
                min_samples_leaf=srt_min_samples_leaf
            )
        super().__init__(_model, *args, **kwargs)

    def _plot_tree(self, model: sklearn.tree.DecisionTreeRegressor, x: pd.DataFrame, report_opt: ReportOptions):
        from matplotlib_setup import plt
        plt.figure(figsize=(min((2 ** model.get_depth()) * 4, 2 ** 15 / 100.0), model.get_depth() * 4))
        sklearn.tree.plot_tree(model, feature_names=self.required_input_data, filled=True, fontsize=12, proportion=True)
        plt.savefig(os.path.join(report_opt.directory, "regression_tree.png"))
        plt.savefig(os.path.join(report_opt.directory, "regression_tree.svg"))
        plt.close()

        with open(os.path.join(report_opt.directory, "regression_tree.txt"), 'wt') as text_file:
            print(sklearn.tree.export_text(model, feature_names=self.required_input_data), file=text_file)

    def _fit_on_model(self, model: sklearn.tree.DecisionTreeRegressor, x: pd.DataFrame, y: pd.Series,
                      report_options: typing.Optional[ReportOptions] = None) -> FitResults:
        model.fit(x, y)
        y_pred = self._predict_on_model(model, x)
        metric_values = {m: getattr(models.sklearn.metrics, m)(y, y_pred) for m in models.sklearn.metrics.__all__}
        res = FitResults(metrics_values=metric_values)

        if report_options is not None:
            self._create_report(model, x, y, y_pred, report_options, res)
            self._plot_tree(model, x, report_options)

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
