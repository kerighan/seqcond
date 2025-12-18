# SeqCond: Sequence Condenser for Autoregressive Models

A TensorFlow library implementing custom attention mechanisms for sequence condensation in autoregressive generation tasks.

## What is SeqCond?

**SeqCond** stands for **Sequence Condenser** - a library designed to condense and efficiently process sequences in autoregressive models. The name reflects our focus on compressing and optimizing sequence representations while maintaining high quality generation.

## Overview

SeqCond provides specialized layers and utilities for building efficient autoregressive models:

- **Custom attention mechanisms** for sequence condensation
- **Memory-efficient architectures** for long sequences
- **Specialized metrics** for language modeling evaluation
- **Modular components** for flexible model construction

## Installation

Install from source (not yet available on PyPI):

```bash
git clone https://github.com/kerighan/seqcond.git
cd seqcond
pip install -e .
```

## Requirements

- Python 3.8+
- TensorFlow 2.0+
- NumPy 1.19.0+

## Core Components

### Attention Layers

- **SeqCondAttention**: Custom attention mechanism for sequence conditioning
- **SeqCondBlock**: Complete transformer block with sequence conditioning
- **RotaryPositionalEmbedding**: Efficient positional encoding

### Normalization & Utilities

- **RMSNorm**: Root Mean Square normalization layer
- **WeightTiedDense**: Weight-tied dense layer for efficient parameter sharing

### Metrics

- **Perplexity**: Standard language modeling metric
- **SparseCategoricalAccuracyIgnoreZero**: Accuracy ignoring padding tokens
- **SparseTopKCategoricalAccuracyIgnoreZero**: Top-K accuracy ignoring padding

## Quick Start

```python
from seqcond.tensorflow import create_seqcond_model

# Create a basic sequence conditioning model
model = create_seqcond_model(
    d_model=512,               # Embedding dimension
    d_ff=2048,                 # Feed-forward dimension
    num_layers=6,              # Number of layers
    vocab_size=10000,          # Vocabulary size
    maxlen=512,                # Maximum sequence length
    num_heads=8,               # Number of attention heads
    seqcond_ratio=2,           # Ratio of SeqCond blocks
    num_thetas=4,              # Number of thetas
    dropout=0.1                # Dropout rate
)

# Compile with appropriate loss
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Model Creation

### Basic Usage

```python
from seqcond.tensorflow import create_seqcond_model

model = create_seqcond_model(
    d_model=768,               # Embedding dimension
    d_ff=3072,                 # Feed-forward dimension
    num_layers=12,             # Number of layers
    vocab_size=50257,          # Vocabulary size
    maxlen=1024,               # Maximum sequence length
    num_heads=12,              # Number of attention heads
    seqcond_ratio=3,           # Ratio of SeqCond blocks
    num_thetas=4,              # Number of thetas for attention
    dropout=0.1,               # Dropout rate
    tie_weights=True,          # Tie embedding weights
    use_conv=True,             # Use convolution in SeqCond blocks
    conv_kernel_size=4         # Convolution kernel size
)
```

### Training Configuration

```python
from seqcond.tensorflow import compile_lm

model = compile_lm(
    model,
    optimizer='adamw',
    learning_rate=3e-4,
    metrics=['accuracy']
)

# Train on your dataset
model.fit(train_dataset, epochs=10, batch_size=32)
```

## Architecture

The library provides building blocks for constructing efficient autoregressive models:

- **Attention mechanisms** designed for sequence condensation
- **Efficient normalization** layers (RMSNorm)
- **Positional encoding** schemes (rotary embeddings)
- **Custom metrics** for proper evaluation

## Applications

- Language modeling
- Text generation
- Sequence-to-sequence tasks
- Any autoregressive generation task

## Development

### Install for development

```bash
git clone https://github.com/kerighan/seqcond.git
cd seqcond
pip install -e .
```

### Run tests

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations:
- **Maixent Chenebaux** - max.chbx@gmail.com
- GitHub: [kerighan](https://github.com/kerighan)

## Status

This is an active research project. The API may evolve as we refine the architecture and add new features.
