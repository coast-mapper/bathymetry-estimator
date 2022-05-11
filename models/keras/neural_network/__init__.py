import argparse
import typing

import argparse_types
from models.abstract import ModelClassMetaData
from models.keras import KerasModel, KerasModelMeta

# Import bez importu :)
if False:
    import tensorflow.keras as keras


class NeuralNetworkMeta(KerasModelMeta):

    def define_parameters(cls, parser: argparse.ArgumentParser):
        if cls == NeuralNetworkModel:
            group = parser.add_argument_group("Neural Network")
            group.add_argument("--nn-layer", required=False,
                               type=argparse_types.separated_tuple([int, str]),
                               help="Neural Network Layer definition", action='append')

    def _refine_configuration(cls, configuration: typing.Dict[str, typing.Any]):
        conf = dict(**super()._refine_configuration(configuration))

        if 'nn_layer' in conf:
            del conf['nn-layer']

        return conf

    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        base = super().model_class_meta_data.__dict__
        base['parameter_prefixes'] = [*base['parameter_prefixes'], 'nn_']
        return ModelClassMetaData(**base)


class NeuralNetworkModel(KerasModel, metaclass=NeuralNetworkMeta):

    def __init__(self,
                 nn_layer: typing.Optional[typing.List[typing.Tuple[int, str]]] = None,
                 *args, **kwargs):
        if nn_layer is not None:
            kwargs['_model'] = self._create_model(nn_layer)
        super().__init__(*args, **kwargs)

    def _compile_model(self, model: 'keras.Model'):
        import tensorflow.keras as keras
        import models.keras.metrics as metrics
        #
        # class SpyOptimizer(keras.optimizers.Adam):
        #
        #
        #     def get_updates(self, loss, params):
        #         ret = super().get_updates(loss, params)
        #         K.print_tensor(ret, 'Get updates')
        #         return ret
        #
        #     def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        #         K.print_tensor(grads_and_vars[0], 'Apply grad')
        #         return super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss='mse', metrics=['mae',
                                           'mse',
                                           metrics.rmse,
                                           metrics.diff_std,
                                           metrics.mpe,
                                           metrics.correlation,
                                           metrics.r_2])

    def _create_model(self, layers_definition: typing.Optional[typing.List[typing.Tuple[int, str]]]) -> 'keras.Model':
        import tensorflow.keras as keras
        from tensorflow.keras import regularizers

        if layers_definition is None:
            raise RuntimeError("Layers are not defined. Please use --nn-layer option.")

        layers = []
        layers_count = len(layers_definition)

        for i in range(layers_count):
            size, fun = layers_definition[i]
            if i == layers_count - 1 and size != 1:
                print("Overriding last layer size to 1")
                size = 1

            layers.append(keras.layers.Dense(size, activation=fun,
                                             kernel_regularizer=regularizers.l2(0.01),
                                             activity_regularizer=regularizers.l2(0.01)))

        return keras.Sequential(layers)
