import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.initializers as initializers


class LinearLayer(keras.layers.Layer):

    def build(self, input_shape):
        self._a = self.add_weight(name="A", shape=[input_shape[1] if len(input_shape) >= 2 else 1],
                                  initializer=initializers.constant(1.0))
        self._b = self.add_weight(name="B", shape=[1],
                                  initializer=initializers.constant(0.0))

    def call(self, inputs, **kwargs):
        mul = self._a * inputs
        if len(inputs.shape) >= 2:
            mul_sum = K.sum(mul, axis=1)
        else:
            mul_sum = mul

        return K.expand_dims(mul_sum + self._b)
