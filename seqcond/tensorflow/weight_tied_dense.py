import tensorflow as tf


class WeightTiedDense(tf.keras.layers.Layer):
    def __init__(self, embedding_layer, use_bias=False, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.use_bias = bool(use_bias)

    def build(self, input_shape):
        self.shared_kernel = self.embedding_layer.embeddings
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.shared_kernel.shape[0],),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    @tf.function(jit_compile=True)
    def call(self, inputs, mask=None):
        inputs = tf.cast(inputs, self.compute_dtype)
        kernel = tf.cast(self.shared_kernel, self.compute_dtype)
        logits = tf.matmul(inputs, kernel, transpose_b=True)
        if self.use_bias:
            logits = tf.nn.bias_add(logits, tf.cast(self.bias, logits.dtype))
        return tf.cast(logits, self.dtype)
