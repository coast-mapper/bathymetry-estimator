import abc
import inspect
import argparse
import dataclasses
import typing

from osgeo import osr
import pandas as pd

__all__ = ['AbstractModelMeta',
           'AbstractModel',
           'ModelClassMetaData',
           'ModelResults',
           'FitResults',
           'EvaluateResults',
           'input_data',
           'ReportOptions']

from scope_utils import Extent


class input_data(abc.ABC):
    x = 'x'
    y = 'y'
    B2 = 'B2'
    B3 = 'B3'
    B4 = 'B4'
    B8 = 'B8'
    raw_bathymetry = 'raw_bathymetry'


@dataclasses.dataclass(frozen=True)
class ModelClassMetaData(object):
    required_input_data: typing.Union[type(Ellipsis), typing.List[typing.Union[str, type(Ellipsis)]]] = ...
    parameter_prefixes: typing.List[str] = dataclasses.field(default_factory=list)
    supports_validation_data: bool = False

    def __post_init__(self):
        if isinstance(self.required_input_data, list):
            ellipsis_count = self.required_input_data.count(Ellipsis)

            if ellipsis_count > 1:
                raise ValueError("required_input_data can contain only one Ellipsis (...)")

    @property
    def required_input_data_has_ellipsis(self):
        return self.required_input_data is Ellipsis or self.required_input_data.count(Ellipsis) > 0


@dataclasses.dataclass
class ModelResults(object):
    metrics_values: typing.Dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class FitResults(ModelResults):
    history: typing.Optional[pd.DataFrame] = None


class EvaluateResults(ModelResults):
    pass


class AbstractModelMeta(abc.ABCMeta):

    def define_parameters(cls, parser: argparse.ArgumentParser):
        pass

    @property
    @abc.abstractmethod
    def model_class_meta_data(cls) -> ModelClassMetaData:
        pass

    @abc.abstractmethod
    def _load_model(cls, path: str, configuration: typing.Dict[str, typing.Any]) -> 'AbstractModel':
        pass

    def get_implementations(cls) -> typing.Set['AbstractModelMeta']:
        sub_classes = set(cls.__subclasses__())

        sub_sub_classes = set()

        for c in sub_classes:
            sub_sub_classes = sub_sub_classes.union(c.get_implementations())

        ret = sub_classes.union(sub_sub_classes)
        if not inspect.isabstract(cls):
            ret.add(cls)
        return ret


@dataclasses.dataclass
class ReportOptions(object):
    directory: str
    label: str
    date: typing.Optional[str] = None


class AbstractModel(object, metaclass=AbstractModelMeta):

    def __init__(self, _input_data_stats: typing.Optional[pd.DataFrame] = None):
        self.__input_data_stats = _input_data_stats

    def save_model(self, path: str):
        import os
        import json
        os.makedirs(path, exist_ok=True)

        meta = {'model': self.__class__.__name__, 'config': self.get_config()}

        with open(os.path.join(path, 'meta.json'), mode='w') as f:
            json.dump(meta, f)

    def get_config(self) -> typing.Dict[str, typing.Any]:
        return {'_input_data_stats': self.__input_data_stats.to_dict()} if self.__input_data_stats is not None else {}

    @property
    def required_input_data(self) -> typing.List[str]:
        if self.__class__.model_class_meta_data.required_input_data_has_ellipsis:
            raise NotImplementedError("This model (%s) has free input data spec. This method should be overridden.")
        ret = self.__class__.model_class_meta_data.required_input_data
        return ret

    @required_input_data.setter
    def required_input_data(self, value: typing.List[str]):
        # NOP :)
        pass

    @property
    def requires_input_data_normalization(self) -> bool:
        return self.__input_data_stats is not None

    @property
    def input_data_stats(self) -> typing.Optional[pd.DataFrame]:
        return self.__input_data_stats

    @input_data_stats.setter
    def input_data_stats(self, input_data_stats: pd.DataFrame):
        self.__input_data_stats = input_data_stats

    @property
    def srs(self) -> typing.Optional[osr.SpatialReference]:
        return None

    @srs.setter
    def srs(self, value: osr.SpatialReference):
        # NOP ;)
        pass

    @property
    def model_extent(self) -> typing.Optional[Extent]:
        return None

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series,
            validation_data: typing.Optional[typing.Tuple[pd.DataFrame, pd.Series]] = None,
            report_options: typing.Optional[ReportOptions] = None
            ) -> FitResults:
        pass

    @abc.abstractmethod
    def evaluate(self, x: pd.DataFrame, y: pd.Series,
                 report_options: typing.Optional[ReportOptions] = None) -> EvaluateResults:
        pass

    @abc.abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.Series:
        pass

    @classmethod
    def load_model(cls, path: str) -> 'AbstractModel':
        import os
        import json

        if not os.path.isdir(path):
            raise NotADirectoryError(path)

        with open(os.path.join(path, 'meta.json'), mode='r') as f:
            meta = json.load(f)
            model_cls_name = meta['model']
            res = list(filter(lambda c: c.__name__ == model_cls_name, cls.get_implementations()))

            if len(res) == 0:
                raise Exception("No such model implementation loaded: %s" % model_cls_name)

            model_cls: 'AbstractModelMeta' = res[0]

            config = meta['config']

            if '_input_data_stats' in config:
                config['_input_data_stats'] = pd.DataFrame.from_dict(config['_input_data_stats'])

            return model_cls._load_model(path, config)
