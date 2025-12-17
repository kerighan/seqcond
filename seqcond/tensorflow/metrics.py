import tensorflow as tf


class SparseCategoricalAccuracyIgnoreZero(tf.keras.metrics.Metric):
    def __init__(self, ignore_class=0, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_class = int(ignore_class)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.ignore_class)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_classes = tf.cast(y_pred_classes, y_true.dtype)

        correct = tf.cast(tf.equal(y_true, y_pred_classes), tf.float32)
        masked_correct = tf.boolean_mask(correct, mask)

        self.total.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))
        self.count.assign_add(tf.reduce_sum(masked_correct))

    @tf.function
    def result(self):
        return tf.cond(
            tf.greater(self.total, 0.0),
            lambda: self.count / self.total,
            lambda: tf.constant(0.0, dtype=self.count.dtype),
        )

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class SparseWeightedCategoricalAccuracyIgnoreZero(tf.keras.metrics.Metric):
    def __init__(
        self,
        ignore_class=0,
        weighting_method="linear",
        name="weighted_accuracy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ignore_class = int(ignore_class)
        self.weighting_method = weighting_method
        self.weighted_total = self.add_weight(
            name="weighted_total", initializer="zeros"
        )
        self.weighted_count = self.add_weight(
            name="weighted_count", initializer="zeros"
        )

    def _compute_position_weights(self, sequence_length):
        positions = tf.range(1, sequence_length + 1, dtype=tf.float32)
        sequence_length_float = tf.cast(sequence_length, tf.float32)

        if self.weighting_method == "linear":
            weights = positions / sequence_length_float
        elif self.weighting_method == "quadratic":
            weights = tf.square(positions / sequence_length_float)
        elif self.weighting_method == "exponential":
            weights = tf.exp(positions / sequence_length_float) - 1.0
            weights = weights / tf.reduce_max(weights)
        else:
            weights = positions / sequence_length_float

        return weights

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.ignore_class)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_classes = tf.cast(y_pred_classes, y_true.dtype)

        correct = tf.cast(tf.equal(y_true, y_pred_classes), tf.float32)
        masked_correct = correct * tf.cast(mask, tf.float32)

        sequence_length = tf.shape(y_true)[1]
        position_weights = self._compute_position_weights(sequence_length)

        batch_size = tf.shape(y_true)[0]
        position_weights = tf.tile(tf.expand_dims(position_weights, 0), [batch_size, 1])

        weighted_correct = masked_correct * position_weights
        weighted_mask = tf.cast(mask, tf.float32) * position_weights

        self.weighted_total.assign_add(tf.reduce_sum(weighted_mask))
        self.weighted_count.assign_add(tf.reduce_sum(weighted_correct))

    @tf.function
    def result(self):
        return tf.cond(
            tf.greater(self.weighted_total, 0.0),
            lambda: self.weighted_count / self.weighted_total,
            lambda: tf.constant(0.0, dtype=self.weighted_count.dtype),
        )

    def reset_state(self):
        self.weighted_total.assign(0.0)
        self.weighted_count.assign(0.0)


class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, ignore_class=0, name="perplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_class = int(ignore_class)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        mask = tf.not_equal(y_true, self.ignore_class)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_masked, y_pred_masked, from_logits=True
        )
        loss_sum = tf.reduce_sum(tf.cast(loss, self.total_loss.dtype))
        self.total_loss.assign_add(loss_sum)
        self.count.assign_add(tf.cast(tf.size(loss), self.count.dtype))

    def result(self):
        mean_loss = tf.math.divide_no_nan(self.total_loss, self.count)
        ppx = tf.exp(mean_loss)
        ppx = tf.clip_by_value(ppx, 0.0, 10000.0)
        return tf.round(ppx * 100.0) / 100.0

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.count.assign(0.0)


class SparseTopKCategoricalAccuracyIgnoreZero(tf.keras.metrics.Metric):
    def __init__(self, k=5, ignore_class=0, name="top_k_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = int(k)
        self.ignore_class = int(ignore_class)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.ignore_class)
        _, top_k_indices = tf.math.top_k(y_pred, k=self.k)
        y_true_reshaped = tf.expand_dims(y_true, -1)

        correct = tf.reduce_any(
            tf.equal(tf.cast(y_true_reshaped, top_k_indices.dtype), top_k_indices),
            axis=-1,
        )
        correct = tf.cast(correct, tf.float32)
        masked_correct = tf.boolean_mask(correct, mask)

        self.total.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))
        self.count.assign_add(tf.reduce_sum(masked_correct))

    @tf.function
    def result(self):
        return tf.cond(
            tf.greater(self.total, 0.0),
            lambda: self.count / self.total,
            lambda: tf.constant(0.0, dtype=self.count.dtype),
        )

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
