"""
Utility functions shared between TensorFlow and JAX implementations.
"""

from .config import Config, ModelConfig


def generate_model_name(config: Config | ModelConfig) -> str:
    """
    Generate a descriptive name for a model based on its configuration.

    Args:
        config: Either a Config or ModelConfig object

    Returns:
        A string name like "seqcond-l12-d768-th8-sh32-m4-r5-o0-a0"

    Note:
        This is a convenience wrapper. You can also use config.name directly.
    """
    return config.name
