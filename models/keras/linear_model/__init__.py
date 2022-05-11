import argparse
import typing

import models.keras

# Import bez importu :)
if False:
    import tensorflow.keras as keras


class LinearModelMeta(models.keras.KerasModelMeta):
    def define_parameters(cls, parser: argparse.ArgumentParser):
        pass

    @property
    def custom_objects(cls) -> typing.Dict[str, typing.Any]:
        from .layer import LinearLayer
        base = super().custom_objects
        base['LinearLayer'] = LinearLayer
        return base


class LinearModel(models.keras.KerasModel, metaclass=LinearModelMeta):

    def __init__(self, _model: typing.Optional['keras.models.Model'] = None, *args, **kwargs):
        if _model is None:
            import tensorflow.keras as keras
            from .layer import LinearLayer
            _model = keras.Sequential([LinearLayer()])

        super().__init__(*args, _model=_model, **kwargs)

    def _compile_model(self, model):
        model.compile(loss='mse')