import tensorflow.keras as keras
import typing
from tensorflow.keras import backend as K


class WeightLayer(keras.layers.Layer):

    def __init__(self, x: float, y: float, r: float,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self._x = x
        self._y = y
        self._r = r

    def get_config(self):
        config = super().get_config()
        config['x'] = self._x
        config['y'] = self._y
        config['r'] = self._r

        return config

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def r(self):
        return self._r

    def call(self, inputs):
        x = inputs[:, 0]
        y = inputs[:, 1]

        dist = K.sqrt(K.pow(x - self._x, 2) + K.pow(y - self._y, 2))

        w = K.pow(K.abs(1 - K.pow(dist / self._r, 2)), 2)
        in_range = K.less(dist, self._r)

        ret = K.expand_dims(K.switch(in_range, w, K.zeros_like(dist)))
        return ret


class GWMKeras(keras.models.Model):
    def __init__(self,
                 model_centers: typing.Optional[typing.List[typing.Tuple[float, float, float]]] = None,
                 model_creator: typing.Optional[typing.Callable[[], keras.layers.Layer]] = None,
                 *args, **kwargs):
        #Drop depricated parameter
        if 'layers_arrangement' in kwargs:
            del kwargs['layers_arrangement']
        super().__init__(*args, **kwargs)

        self.__models_and_weights = None
        if model_centers is not None and model_creator is not None:
            self.__models_and_weights = []
            for x_c, y_c, r in model_centers:
                m_layer = model_creator()
                w_layer = WeightLayer(x_c, y_c, r)
                self.models_and_weights.append((m_layer, w_layer))




    @property
    def models_and_weights(self):
        if self.__models_and_weights is None:
            raise RuntimeError("__models_and_weights is not defined")
        return self.__models_and_weights

    @models_and_weights.setter
    def models_and_weights(self, value):
        self.__models_and_weights = value

    @property
    def centers(self) -> typing.List[typing.Tuple[float,float,float]]:
        return [(wl.x,wl.y,wl.r) for _, wl in self.models_and_weights]

    def call(self, inputs, training=None, mask=None):
        xy = inputs[:, 0:2]
        rest_of_inputs = inputs[:, 2:]

        results = []
        weights = []

        for m_layer, w_layer in self.models_and_weights:
            w = w_layer(xy)
            res = m_layer(rest_of_inputs) * w

            results.append(res)
            weights.append(w)

        re_cat = K.concatenate(results)
        re = K.sum(re_cat, axis=-1)
        we_cat = K.concatenate(weights)
        we = K.sum(we_cat, axis=-1)

        fin_res = K.switch(K.greater(we, 0), re / we, K.zeros_like(we))
        return fin_res

    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(**config)
