import argparse
import typing

import argparse_types
from models.abstract import ModelClassMetaData, input_data
from models.keras.geography_weighted_model import GeographyWeightedKerasModelMeta, GeographyWeightedKerasModel

# Import bez importu :)
if False:
    import tensorflow.keras as keras


class GeographyWeightedNeuralNetworkModelMeta(GeographyWeightedKerasModelMeta):

    def define_parameters(cls, parser: argparse.ArgumentParser):
        if cls == GeographyWeightedNeuralNetworkModel:
            group = parser.add_argument_group("Geography Weighted Model Neural Network")
            group.add_argument("--gwmknn-layer", required=False,
                               type=argparse_types.separated_tuple([int, str]),
                               help="Neural Network Layer definition", action='append')

    def _refine_configuration(cls, configuration: typing.Dict[str, typing.Any]):
        conf = dict(**super()._refine_configuration(configuration))

        if 'gwmknn_layer' in conf:
            del conf['gwmknn-layer']

        return conf

    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        base = super().model_class_meta_data.__dict__
        base['parameter_prefixes'] = [*base['parameter_prefixes'], 'gwmknn_']
        return ModelClassMetaData(**base)


class GeographyWeightedNeuralNetworkModel(GeographyWeightedKerasModel,
                                          metaclass=GeographyWeightedNeuralNetworkModelMeta):

    def __init__(self, gwmknn_layer: typing.Optional[typing.List[typing.Tuple[int, str]]] = None,
                 *args, **kwargs):
        self.__layers_definition = gwmknn_layer
        super().__init__(*args, **kwargs)

    def _compile_model(self, model: 'keras.Model'):
        import tensorflow.keras as keras
        import models.keras.metrics as metrics
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=metrics.rmse, metrics=['mae',
                                                  'mse',
                                                  metrics.rmse,
                                                  metrics.diff_std,
                                                  metrics.mpe,
                                                  metrics.correlation,
                                                  metrics.r_2])

    def _create_submodel(self) -> 'keras.Model':
        import tensorflow.keras as keras

        if self.__layers_definition is None:
            raise RuntimeError("Layers are not defined. Please use --gwmknn-layer option.")

        layers = []
        layers_count = len(self.__layers_definition)

        for i in range(layers_count):
            size, fun = self.__layers_definition[i]
            if i == layers_count-1 and size != 1:
                print("Overriding last layer size to 1")
                size = 1

            layers.append(keras.layers.Dense(size, activation=fun))

        return keras.Sequential(layers)
