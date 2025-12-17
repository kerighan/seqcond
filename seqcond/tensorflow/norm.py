import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        x_f32 = tf.cast(x, tf.float32)
        mean_sq = tf.reduce_mean(tf.square(x_f32), axis=-1, keepdims=True)
        y = x_f32 * tf.math.rsqrt(mean_sq + self.epsilon)
        y = y * tf.cast(self.scale, y.dtype)
        return tf.cast(y, x.dtype)
