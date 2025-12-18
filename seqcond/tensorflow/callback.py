import tensorflow as tf
import numpy as np
import os


class StepwiseGenerationCallback(tf.keras.callbacks.Callback):
    """
    Callback that triggers autoregressive text generation every n steps.

    This callback allows for sampling from the model during training to monitor
    generation quality at regular intervals.
    """

    def __init__(
        self,
        trigger_every_n_steps=100,
        nlp=None,
        prompt="<|im_start|>user\n",
        max_new_tokens=64,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=2,
        seed=None,
        model_name=None,
    ):
        """
        Initialize the callback.

        Args:
            trigger_every_n_steps: Number of steps between generations
            prompt: Initial text to start generation from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep for sampling
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize)
            seed: Random seed for reproducibility
            model_name: Name of the model for logging generations to file
        """
        super().__init__()
        self.trigger_every_n_steps = trigger_every_n_steps
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.step_counter = 0
        self.nlp = nlp
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
            # Reset file by opening in write mode
            with open(self.log_file, "w") as f:
                f.write(f"=== Generations for {self.model_name} ===\n\n")

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        self.step_counter += 1

        if self.step_counter % self.trigger_every_n_steps == 0:
            self.generate_text()

    def _apply_top_k(self, probs):
        if not (self.top_k > 0 and self.top_k < probs.size):
            return probs
        top_k_indices = np.argpartition(probs, -self.top_k)[-self.top_k :]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_k_indices] = True
        probs = probs.copy()
        probs[~mask] = 0.0
        s = probs.sum()
        return probs / s if s > 0 else probs

    def _apply_top_p(self, probs):
        if self.top_p >= 1.0:
            s = probs.sum()
            return probs / s if s > 0 else probs

        if self.top_p <= 0.0:
            idx = int(np.argmax(probs))
            out = np.zeros_like(probs)
            out[idx] = 1.0
            return out

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_indices = sorted_indices[cumulative_probs <= self.top_p]
        if cutoff_indices.size == 0:
            cutoff_indices = sorted_indices[:1]

        mask = np.zeros_like(probs, dtype=bool)
        mask[cutoff_indices] = True
        probs = probs.copy()
        probs[~mask] = 0.0

        s = probs.sum()
        if s == 0:
            probs[sorted_indices[0]] = 1.0
            return probs
        return probs / s

    def sample_from_logits(self, logits, generated_tokens=None):
        logits = logits.astype(np.float64, copy=True)

        if generated_tokens and self.repetition_penalty != 1.0:
            idx = np.fromiter(set(generated_tokens), dtype=np.int64)
            idx = idx[idx < logits.size]
            if idx.size:
                pos = logits[idx] > 0
                logits[idx[pos]] /= self.repetition_penalty
                logits[idx[~pos]] *= self.repetition_penalty

        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        probs = self._apply_top_k(probs)
        probs = self._apply_top_p(probs)
        return int(np.random.choice(probs.size, p=probs))

    def generate_text(self):
        """Generate text using autoregressive sampling with top-k, temperature, and repetition penalty."""
        print(f"\n--- Generation at step {self.step_counter} ---")
        txt = self.prompt
        tokens = self.nlp([txt])[0]
        generated_tokens = []

        for _ in range(self.max_new_tokens):
            logits = self.model.predict(np.array([tokens], dtype=np.int32), verbose=0)
            token_id = self.sample_from_logits(
                logits[0, len(tokens) - 1], generated_tokens
            )
            token = self.nlp.decode([token_id])
            tokens.append(token_id)
            generated_tokens.append(token_id)
            txt += token
            print(token, end="", flush=True)

        print()

        # Write to file if model_name is set
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"--- Step {self.step_counter} ---\n")
                f.write(txt + "\n\n")
