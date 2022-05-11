import abc
import argparse
import os
import pickle
import typing

import pandas as pd
import sklearn

import models.abstract
from models.abstract import ModelClassMetaData, ReportOptions, EvaluateResults, FitResults


class SklearnModelMeta(models.abstract.AbstractModelMeta):
    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        return ModelClassMetaData(parameter_prefixes=['sklearn_'])

    def define_parameters(cls, parser: argparse.ArgumentParser):
        if cls == SklearnModel:
            group = parser.add_argument_group("Sklearn Engine options")
            group.add_argument("--sklearn-backend", choices=['loky', 'threading'], default='loky')

    def _refine_configuration(cls, configuration: typing.Dict[str, typing.Any]):
        if '_model' in configuration:
            del configuration['_model']
        return configuration

    def _load_model(cls, path: str, configuration: typing.Dict[str, typing.Any]) -> 'AbstractModel':
        configuration = cls._refine_configuration(configuration)

        model_file_path = os.path.join(path, "model.pickle")

        with open(model_file_path, 'rb') as model_file:
            sklern_model = pickle.load(model_file)

        configuration['_model'] = sklern_model

        return cls(**configuration)


_M = typing.TypeVar('_M')


class SklearnModel(models.abstract.AbstractModel, typing.Generic[_M], metaclass=SklearnModelMeta):
    def __init__(self, _model: _M,
                 _required_input_data: typing.Optional[typing.List[str]] = None,
                 sklearn_backend: str = 'loky',
                 *args,**kwargs):
        super().__init__(*args,**kwargs)
        self._model = _model
        self.__required_input_data = _required_input_data
        self.__backend = sklearn_backend

    @property
    def model(self) -> _M:
        return self._model

    @property
    def required_input_data(self) -> typing.List[str]:
        return self.__required_input_data

    @required_input_data.setter
    def required_input_data(self, value: typing.List[str]):
        self.__required_input_data = value

    def save_model(self, path: str):
        super().save_model(path)
        model_file_path = os.path.join(path, "model.pickle")

        with open(model_file_path, 'wb') as model_file:
            pickle.dump(self._model, model_file)

    def get_config(self) -> typing.Dict[str, typing.Any]:
        return {
            **super().get_config(),
            'sklearn_backend': self.__backend,
            '_required_input_data': self.__required_input_data
        }

    def _create_report(self,
                       model: _M,
                       x: pd.DataFrame,
                       y: pd.Series,
                       y_pred: pd.Series,
                       report_opt: ReportOptions,
                       model_results: typing.Union[models.abstract.ModelResults,
                                                   models.abstract.EvaluateResults,
                                                   models.abstract.FitResults]):
        import os
        from matplotlib import pyplot as plt
        from plots import fit_scatter

        os.makedirs(report_opt.directory, exist_ok=True)

        report = open(os.path.join(report_opt.directory, '%s_report.txt' % report_opt.label), "w")
        with report:
            print(model_results.metrics_values, file=report)

        fig = fit_scatter(y, y_pred, "%s data [m]" % report_opt.label, "%s prediction [m]" % report_opt.label,
                          report_opt.date)
        plt.savefig(os.path.join(report_opt.directory, "%s_scatter.png" % report_opt.label))
        plt.savefig(os.path.join(report_opt.directory, "%s_scatter.svg" % report_opt.label))
        plt.close(fig)
        fig = fit_scatter(y, y_pred, "%s data [m]" % report_opt.label, "%s prediction [m]" % report_opt.label,
                          report_opt.date,
                          without_description=True)
        plt.savefig(os.path.join(report_opt.directory, "%s_scatter_raw.png" % report_opt.label))
        plt.savefig(os.path.join(report_opt.directory, "%s_scatter_raw.svg" % report_opt.label))
        plt.close(fig)

    @abc.abstractmethod
    def _fit_on_model(self, model: _M,
                      x: pd.DataFrame, y: pd.Series,
                      report_options: typing.Optional[ReportOptions] = None) -> FitResults:
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series,
            validation_data: typing.Optional[typing.Tuple[pd.DataFrame, pd.Series]] = None,
            report_options: typing.Optional[ReportOptions] = None) -> FitResults:
        with sklearn.utils.parallel_backend(self.__backend):
            return self._fit_on_model(self._model, x, y, report_options)

    @abc.abstractmethod
    def _evaluate_on_model(self, model: _M, x: pd.DataFrame, y: pd.Series,
                           report_options: typing.Optional[ReportOptions] = None) -> EvaluateResults:
        pass

    def evaluate(self, x: pd.DataFrame, y: pd.Series,
                 report_options: typing.Optional[ReportOptions] = None) -> EvaluateResults:
        with sklearn.utils.parallel_backend(self.__backend):
            return self._evaluate_on_model(self._model, x, y, report_options)

    @abc.abstractmethod
    def _predict_on_model(self, model: _M, x: pd.DataFrame) -> pd.Series:
        pass

    def predict(self, x: pd.DataFrame) -> pd.Series:
        with sklearn.utils.parallel_backend(self.__backend):
            return self._predict_on_model(self._model, x)
