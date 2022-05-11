import argparse
import typing
import os
import abc

import pandas as pd

import models.abstract
from models.abstract import ModelClassMetaData, ReportOptions, EvaluateResults, FitResults

# Import bez importu :)
if False:
    import tensorflow.keras as keras


class KerasModelMeta(models.abstract.AbstractModelMeta):

    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        """
        Metadane modeli opartych na Kerasie.
        Tę metodę należy nadpisać i do tego co zwraca dodać prefiksy parametrów modelu dziedziczącego (jeśli takie są)
        oraz ewentualnie ustawić wymagane dane wejściowe na konkretene wartości.
        """
        return ModelClassMetaData(
            required_input_data=...,
            parameter_prefixes=['keras_'],
            supports_validation_data=True
        )

    def define_parameters(cls, parser: argparse.ArgumentParser):
        """
        Tę metodę należy nadpisać nawet jeśli nie ma się tam nic ciekawego do zaimplementowania.
        Najlepiej (pomimo zabezpieczenia) nie wywoływać jej z klas pochodnych.
        """
        if cls == KerasModel:
            group = parser.add_argument_group("Keras Engine")
            group.add_argument("--keras-max-iterations", type=int, default=1000,
                               help="Maximal count of train iterations.")
            group.add_argument("--keras-batch-size", type=int, default=1000,
                               help="Count of samples precessed at once by Keras.")
            group.add_argument("--keras-validation-stop", type=int, default=10,
                               help="Count of iteration after training is stopped due to validation loss rise.")
            group.add_argument("--keras-model-checkpoint-dir", type=int, required=False,
                               help="Checkpoint directory in which model weights will be saved during fit procedure.")

    @property
    def custom_objects(cls) -> typing.Dict[str, typing.Any]:
        import models.keras.metrics as metrics

        return {m: getattr(metrics, m) for m in metrics.__all__}

    def _refine_configuration(cls, configuration: typing.Dict[str, typing.Any]):
        if '_model' in configuration:
            del configuration['_model']
        return configuration

    def _load_keras_model(cls, path: str) -> 'keras.models.Model':
        import tensorflow.keras as keras
        model_path = os.path.join(path, 'keras_model.tf')
        keras_model = keras.models.load_model(model_path, custom_objects=cls.custom_objects)
        return keras_model

    def _load_model(cls, path: str, configuration: typing.Dict[str, typing.Any]) -> models.abstract.AbstractModel:
        kears_model = cls._load_keras_model(path)
        configuration = cls._refine_configuration(configuration)
        configuration['_model'] = kears_model

        return cls(**configuration)


class KerasModel(models.abstract.AbstractModel, metaclass=KerasModelMeta):

    def __init__(self,
                 keras_max_iterations: int,
                 keras_batch_size: int,
                 keras_validation_stop: int,
                 keras_model_checkpoint_dir: typing.Optional[str] = None,
                 _required_input_data: typing.List[str] = [],
                 _model: typing.Optional['keras.models.Model'] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__required_input_data = _required_input_data
        self.__keras_max_iterations = keras_max_iterations
        self.__keras_batch_size = keras_batch_size
        self.__keras_validation_stop = keras_validation_stop
        self.__keras_model_checkpoint_dir = keras_model_checkpoint_dir
        self.__model = _model

    @abc.abstractmethod
    def _compile_model(self, model: 'keras.Model'):
        pass

    def _create_report(self,
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

        if issubclass(type(model_results), models.abstract.FitResults):
            fname = os.path.join(report_opt.directory, '%s_report.txt' % report_opt.label)
            model_results.history.to_csv(fname)
        elif issubclass(type(model_results), models.abstract.ModelResults):
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

    def _check_model(self, model: 'keras.Model'):
        if model is None:
            raise RuntimeError("Model is not set")

        if model.compiled_loss is None:
            self._compile_model(model)

    @property
    def model(self) -> 'keras.models.Model':
        return self.__model

    @property
    def max_iterations(self) -> int:
        return self.__keras_max_iterations

    @property
    def batch_size(self) -> int:
        return self.__keras_batch_size

    @property
    def validation_stop(self) -> int:
        return self.__keras_validation_stop

    @property
    def required_input_data(self) -> typing.List[str]:
        return self.__required_input_data

    @required_input_data.setter
    def required_input_data(self, value: typing.List[str]):
        self.__required_input_data = value

    def save_model(self, path: str):
        super().save_model(path)

        model_path = os.path.join(path, 'keras_model.tf')

        self.model.save(model_path)

    def get_config(self) -> typing.Dict[str, typing.Any]:
        return {
            **super().get_config(),
            'keras_max_iterations': self.__keras_max_iterations,
            'keras_batch_size': self.__keras_batch_size,
            'keras_validation_stop': self.__keras_validation_stop,
            '_required_input_data': self.__required_input_data
        }

    def _fit_on_model(self, model: 'keras.Model', x: pd.DataFrame, y: pd.Series,
                      validation_data: typing.Optional[typing.Tuple[pd.DataFrame, pd.Series]] = None,
                      report_options: typing.Optional[ReportOptions] = None,
                      sample_weights: typing.Optional[pd.Series] = None,
                      validation_weights: typing.Optional[pd.Series] = None) -> FitResults:
        import tensorflow.keras as keras
        self._check_model(model)

        callbacks = []

        if validation_data is not None:
            callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=self.__keras_validation_stop))
            if validation_weights is not None:
                validation_data = tuple([*validation_data, validation_weights])

        if report_options is not None:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=report_options.directory, histogram_freq=1))

        if self.__keras_model_checkpoint_dir is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.__keras_model_checkpoint_dir,
                                      'keras_model-{epoch:02d}-{val_loss:.2f}.tf')
            ))

        history: keras.callbacks.History = model.fit(x=x, y=y, validation_data=validation_data,
                                                     batch_size=self.__keras_batch_size,
                                                     epochs=self.__keras_max_iterations,
                                                     callbacks=callbacks,
                                                     sample_weight=sample_weights)
        h_: pd.DataFrame = pd.DataFrame.from_dict(history.history)
        last_row = h_.loc[max(h_.index)].to_dict()
        fit_res = FitResults(last_row, h_)

        if report_options is not None:
            self._create_fit_report(model, x, y, validation_data, fit_res, report_options)

        return fit_res

    def _create_fit_report(self, model, x, y, validation_data, fit_res, report_options):
        y_pred = self._predict_on_model(model, x)
        self._create_report(x, y, y_pred, report_options, fit_res)
        if validation_data is not None:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
            else:
                val_x, val_y, val_w = validation_data
            validation_report = ReportOptions(**report_options.__dict__)
            validation_report.label = 'Validation'

            self._evaluate_on_model(model, val_x, val_y, validation_report)

    def fit(self, x: pd.DataFrame, y: pd.Series,
            validation_data: typing.Optional[typing.Tuple[pd.DataFrame, pd.Series]] = None,
            report_options: typing.Optional[ReportOptions] = None) -> FitResults:
        return self._fit_on_model(self.__model, x, y, validation_data, report_options)

    def _evaluate_on_model(self, model: 'keras.Model', x: pd.DataFrame, y: pd.Series,
                           report_options: typing.Optional[ReportOptions] = None) -> EvaluateResults:
        import tensorflow.keras as keras
        self._check_model(model)
        callbacks = []
        if report_options is not None:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=report_options.directory, histogram_freq=1))

        ev_res = model.evaluate(x, y,
                                batch_size=len(x),
                                callbacks=callbacks)

        if isinstance(ev_res, list):
            ret = EvaluateResults(dict(zip(model.metrics_names, ev_res)))
        else:
            ret = EvaluateResults({'loss': ev_res})

        if report_options is not None:
            y_pred = self._predict_on_model(model, x)
            self._create_report(x, y, y_pred, report_options, ret)
        return ret

    def evaluate(self, x: pd.DataFrame, y: pd.Series,
                 report_options: typing.Optional[ReportOptions] = None) -> EvaluateResults:
        return self._evaluate_on_model(self.__model, x, y, report_options)

    def _predict_on_model(self, model: 'keras.Model', x: pd.DataFrame) -> pd.Series:
        self._check_model(model)
        if len(x) != 0:
            res = model.predict(x, batch_size=self.__keras_batch_size)
            res = pd.Series(res.reshape(len(res)), index=x.index)
            return res
        else:
            import numpy as np
            return pd.Series(np.array([]))

    def predict(self, x: pd.DataFrame) -> pd.Series:
        return self._predict_on_model(self.__model, x)
