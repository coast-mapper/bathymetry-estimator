import typing

import models.abstract
from models.abstract import ModelClassMetaData
from models.keras.geography_weighted_model import GeographyWeightedKerasModelMeta, GeographyWeightedKerasModel

# Import bez importu :)
if False:
    import tensorflow.keras as keras


class GeographyWeightedLinearModelMeta(GeographyWeightedKerasModelMeta):

    @property
    def model_class_meta_data(cls) -> ModelClassMetaData:
        base = super().model_class_meta_data.__dict__
        base['required_input_data'] = [models.abstract.input_data.x,
                                       models.abstract.input_data.y,
                                       models.abstract.input_data.raw_bathymetry]

        return ModelClassMetaData(**base)


class GeographyWeightedLinearModel(GeographyWeightedKerasModel, metaclass=GeographyWeightedLinearModelMeta):
    def _compile_model(self, model: 'keras.Model'):
        import tensorflow.keras as keras
        import models.keras.metrics as metrics
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1, momentum=0.5),
                      loss=metrics.rmse, metrics=['mae',
                                                  'mse',
                                                  metrics.mpe,
                                                  metrics.correlation,
                                                  metrics.r_2],
                      run_eagerly=False)

    def _create_submodel(self) -> 'keras.Model':
        import tensorflow.keras as keras
        from models.keras.linear_model.layer import LinearLayer
        return keras.Sequential(LinearLayer())

    @property
    def required_input_data(self) -> typing.List[str]:
        ret = self.__class__.model_class_meta_data.required_input_data
        return ret


