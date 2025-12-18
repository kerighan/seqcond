import math
import tensorflow as tf

from .rope import RotaryPositionalEmbedding, TransformerDecoderBlock
from .seqcond_fast import SeqCondBlock
from .weight_tied_dense import WeightTiedDense


def create_transformer_model(
    d_model=256,
    d_ff=768,
    num_layers=12,
    num_heads=8,
    num_kv_heads=None,
    vocab_size=100300,
    maxlen=1024,
    dropout=0.0,
    tie_weights=True,
    qk_norm=False,
    qk_norm_eps=1e-6,
):
    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

    embedding = tf.keras.layers.Embedding(
        vocab_size, d_model, mask_zero=True, name="token_embedding"
    )

    x = embedding(inputs)
    mask = embedding.compute_mask(inputs)

    rotary = RotaryPositionalEmbedding(maxlen, d_model // num_heads)
    cos, sin = rotary(inputs, n_heads=num_heads)

    for i in range(num_layers):
        block = TransformerDecoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            name=f"transformer_block_{i}",
        )
        x = block(x, cos=cos, sin=sin, mask=mask)

    if tie_weights:
        logits = WeightTiedDense(
            embedding_layer=embedding, use_bias=False, name="output_projection"
        )(x)
    else:
        logits = tf.keras.layers.Dense(
            vocab_size, use_bias=False, name="output_projection"
        )(x)

    return tf.keras.Model(inputs=inputs, outputs=logits)


def create_seqcond_model(
    d_model=256,
    d_ff=768,
    num_layers=12,
    vocab_size=100300,
    maxlen=1024,
    use_positional_embedding=False,
    seqcond_ratio=3,
    num_heads=8,
    num_kv_heads=None,
    seqcond_heads=None,
    num_anchor_heads=0,
    num_thetas=4,
    derivative_order=0,
    derivative_aggregation="re_im",
    dropout=0.0,
    tie_weights=True,
    qk_norm=False,
    qk_norm_eps=1e-6,
    use_conv=True,
    conv_kernel_size=4,
):
    if seqcond_heads is None:
        seqcond_heads = num_heads

    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

    embedding = tf.keras.layers.Embedding(
        vocab_size, d_model, mask_zero=True, name="token_embedding"
    )

    x = embedding(inputs)
    mask = embedding.compute_mask(inputs)

    if use_positional_embedding:
        if maxlen is None:
            raise ValueError("maxlen must be set when use_positional_embedding=True")
        position_embedding = tf.keras.layers.Embedding(
            maxlen, d_model, name="position_embedding"
        )
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
        x = x + position_embedding(positions)

    rotary = RotaryPositionalEmbedding(maxlen, d_model // num_heads)
    cos, sin = rotary(inputs, n_heads=num_heads)

    transformer_idx = 0
    seqcond_idx = 0
    for i in range(num_layers):
        if (i + 1) % (seqcond_ratio + 1) == 0:
            block = TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                name=f"transformer_block_{transformer_idx}",
            )
            x = block(x, cos=cos, sin=sin, mask=mask)
            transformer_idx += 1
        else:
            block = SeqCondBlock(
                num_heads=seqcond_heads,
                num_thetas=num_thetas,
                num_anchor_heads=num_anchor_heads,
                derivative_order=derivative_order,
                dropout=dropout,
                use_conv=use_conv,
                conv_kernel_size=conv_kernel_size,
                name=f"seqcond_block_{seqcond_idx}",
            )
            x = block(x, mask=mask)
            seqcond_idx += 1

    if tie_weights:
        logits = WeightTiedDense(
            embedding_layer=embedding, use_bias=False, name="output_projection"
        )(x)
    else:
        logits = tf.keras.layers.Dense(
            vocab_size, use_bias=False, name="output_projection"
        )(x)

    return tf.keras.Model(inputs=inputs, outputs=logits)


def compile_lm(
    model,
    base_lr=1e-3,
    warmup_steps=1000,
    total_steps=1_000_000,
    ignore_class=0,
    weight_decay=1e-2,
    clipnorm=1.0,
    beta_1=0.9,
    beta_2=0.999,
):
    try:
        from tensorflow.keras.optimizers import AdamW
    except ImportError:
        from tensorflow.keras.optimizers.experimental import AdamW

    from .metrics import (
        Perplexity,
        SparseCategoricalAccuracyIgnoreZero,
        SparseTopKCategoricalAccuracyIgnoreZero,
        SparseWeightedCategoricalAccuracyIgnoreZero,
    )

    decay_steps = total_steps - warmup_steps

    class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr, warmup_steps, decay_steps, alpha=0.0):
            super().__init__()
            self.base_lr = tf.cast(base_lr, tf.float32)
            self.warmup_steps = tf.cast(warmup_steps, tf.float32)
            self.decay_steps = tf.cast(decay_steps, tf.float32)
            self.alpha = tf.cast(alpha, tf.float32)

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_lr = self.base_lr * (step / self.warmup_steps)
            decay_progress = (step - self.warmup_steps) / self.decay_steps
            decay_progress = tf.clip_by_value(decay_progress, 0.0, 1.0)
            cosine_decay = 0.5 * (
                1.0 + tf.cos(tf.constant(math.pi, tf.float32) * decay_progress)
            )
            decayed_lr = self.alpha + (self.base_lr - self.alpha) * cosine_decay
            return tf.cond(
                step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr
            )

        def get_config(self):
            return {
                "base_lr": self.base_lr,
                "warmup_steps": self.warmup_steps,
                "decay_steps": self.decay_steps,
                "alpha": self.alpha,
            }

    lr_schedule = WarmupCosineDecaySchedule(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        alpha=1e-5,
    )

    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
        beta_1=beta_1,
        beta_2=beta_2,
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, ignore_class=ignore_class
    )

    metrics = [
        SparseCategoricalAccuracyIgnoreZero(ignore_class=ignore_class, name="acc"),
        SparseTopKCategoricalAccuracyIgnoreZero(
            k=5, ignore_class=ignore_class, name="topk"
        ),
        # SparseWeightedCategoricalAccuracyIgnoreZero(
        #     ignore_class=ignore_class, name="long"
        # ),
        Perplexity(ignore_class=ignore_class, name="ppx"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


class GradientAccumulationModel(tf.keras.Model):
    def __init__(self, inner_model, accum_steps=1, **kwargs):
        super().__init__(**kwargs)
        self.inner_model = inner_model
        self.accum_steps = max(1, int(accum_steps))
        self._accum_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._accum_grads = None
        self._has_grad = None

    def call(self, inputs, training=False):
        return self.inner_model(inputs, training=training)

    def _maybe_init_accumulators(self):
        if self._accum_grads is not None:
            return
        if not self.trainable_variables:
            return
        self._accum_grads = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in self.trainable_variables
        ]
        self._has_grad = [False for _ in self.trainable_variables]

    def train_step(self, data):
        if self.accum_steps <= 1:
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                    regularization_losses=self.losses,
                )
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
            results = {m.name: m.result() for m in self.metrics}
            results["loss"] = loss
            return results

        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

            loss_to_backprop = loss / tf.cast(self.accum_steps, loss.dtype)

        gradients = tape.gradient(loss_to_backprop, self.trainable_variables)

        self._maybe_init_accumulators()
        if self._accum_grads is None:
            self._accum_grads = [
                tf.Variable(tf.zeros_like(v), trainable=False)
                for v in self.trainable_variables
            ]
            self._has_grad = [False for _ in self.trainable_variables]

        for acc_g, g in zip(self._accum_grads, gradients):
            if g is None:
                continue
            acc_g.assign_add(g)

        if self._has_grad is not None:
            for i, g in enumerate(gradients):
                if g is not None:
                    self._has_grad[i] = True

        self._accum_step_counter.assign_add(1)

        def _apply_and_reset():
            grads_and_vars = []
            for i, (acc_g, v) in enumerate(
                zip(self._accum_grads, self.trainable_variables)
            ):
                if self._has_grad is None or self._has_grad[i]:
                    grads_and_vars.append((acc_g, v))

            if grads_and_vars:
                self.optimizer.apply_gradients(grads_and_vars)

            for i, acc_g in enumerate(self._accum_grads):
                if self._has_grad is None or self._has_grad[i]:
                    acc_g.assign(tf.zeros_like(acc_g))

            if self._has_grad is not None:
                for i in range(len(self._has_grad)):
                    self._has_grad[i] = False
            self._accum_step_counter.assign(0)
            return tf.no_op()

        tf.cond(
            tf.equal(self._accum_step_counter, self.accum_steps),
            _apply_and_reset,
            lambda: tf.no_op(),
        )

        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss = self.compiled_loss(
            y,
            y_pred,
            sample_weight=sample_weight,
            regularization_losses=self.losses,
        )
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results
