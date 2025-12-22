import tensorflow as tf
import numpy as np
import os

from .generate import generate_text


class StepwiseGenerationCallback(tf.keras.callbacks.Callback):
    """
    Callback that triggers autoregressive text generation every n steps.
    Uses the generate_text function internally.
    """

    def __init__(
        self,
        tokenizer,
        trigger_every_n_steps=100,
        prompt="<|im_start|>user\n",
        max_new_tokens=64,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.5,
        maxlen=768,
        seed=None,
        model_name=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.trigger_every_n_steps = trigger_every_n_steps
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.maxlen = maxlen
        self.seed = seed
        self.step_counter = 0
        self.model_name = model_name
        self.log_file = None

        if self.seed is not None:
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training. Reset the log file."""
        if self.model_name is not None:
            os.makedirs("generations", exist_ok=True)
            self.log_file = f"generations/{self.model_name}.txt"
            with open(self.log_file, "w") as f:
                f.write(f"=== Generations for {self.model_name} ===\n\n")

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        self.step_counter += 1

        if self.step_counter % self.trigger_every_n_steps == 0:
            print(f"\n--- Generation at step {self.step_counter} ---")
            txt = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                maxlen=self.maxlen,
                verbose=True,
            )

            if self.log_file is not None:
                with open(self.log_file, "a") as f:
                    f.write(f"--- Step {self.step_counter} ---\n")
                    f.write(txt + "\n\n")
