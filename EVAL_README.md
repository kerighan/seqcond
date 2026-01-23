# SeqCond Evaluation Guide

This guide covers all available evaluation benchmarks and commands for the SeqCond model.

## Quick Start

```bash
# Basic evaluation on HellaSwag
python eval.py --benchmark hellaswag --split validation

# Evaluate with limited samples
python eval.py --benchmark hellaswag --split validation --max_samples 100
```

## Available Benchmarks

### 1. HellaSwag

Commonsense reasoning benchmark with sentence completion tasks.

```bash
# Validation split (default)
python eval.py --benchmark hellaswag --split validation

# Test split
python eval.py --benchmark hellaswag --split test
```

### 2. GPQA (Graduate-Level Google-Proof Q&A)

Expert-level science questions. **Note: Requires HuggingFace authentication for gated dataset.**

```bash
# GPQA Diamond (198 examples)
python eval.py --benchmark gpqa:diamond

# Limited samples
python eval.py --benchmark gpqa:diamond --max_samples 50
```

### 3. MMLU (Massive Multitask Language Understanding)

Comprehensive benchmark covering 57 subjects across STEM, humanities, social sciences, and more.

```bash
# Evaluate a specific subject
python eval.py --benchmark mmlu:anatomy --split test
python eval.py --benchmark mmlu:college_physics --split test
python eval.py --benchmark mmlu:philosophy --split test

# Evaluate ALL subjects (takes a long time!)
python eval.py --benchmark mmlu --split test
python eval.py --benchmark mmlu:all --split test
```

**Popular MMLU subjects:**
- **STEM**: `abstract_algebra`, `astronomy`, `college_biology`, `college_chemistry`, `college_computer_science`, `college_mathematics`, `college_physics`, `computer_security`, `electrical_engineering`, `high_school_biology`, `high_school_chemistry`, `high_school_computer_science`, `high_school_mathematics`, `high_school_physics`, `machine_learning`
- **Humanities**: `formal_logic`, `high_school_european_history`, `high_school_us_history`, `high_school_world_history`, `international_law`, `jurisprudence`, `logical_fallacies`, `moral_disputes`, `moral_scenarios`, `philosophy`, `prehistory`, `professional_law`, `world_religions`
- **Social Sciences**: `econometrics`, `high_school_geography`, `high_school_government_and_politics`, `high_school_macroeconomics`, `high_school_microeconomics`, `high_school_psychology`, `human_sexuality`, `professional_psychology`, `public_relations`, `security_studies`, `sociology`, `us_foreign_policy`
- **Other**: `anatomy`, `business_ethics`, `clinical_knowledge`, `college_medicine`, `global_facts`, `human_aging`, `management`, `marketing`, `medical_genetics`, `miscellaneous`, `nutrition`, `professional_accounting`, `professional_medicine`, `virology`

### 4. ARC (AI2 Reasoning Challenge)

Science question answering with two difficulty levels.

```bash
# ARC-Easy
python eval.py --benchmark arc:easy --split test

# ARC-Challenge (harder)
python eval.py --benchmark arc:challenge --split test

# Both Easy and Challenge
python eval.py --benchmark arc --split test
```

### 5. PIQA (Physical Interaction QA)

Physical commonsense reasoning about everyday situations.

```bash
# Validation split
python eval.py --benchmark piqa --split validation

# Test split (labels not public)
python eval.py --benchmark piqa --split test
```

### 6. Winogrande

Commonsense reasoning with pronoun resolution tasks.

```bash
# XL version (default, 1267 examples)
python eval.py --benchmark winogrande:xl --split validation

# Large version (1267 examples)
python eval.py --benchmark winogrande:l --split validation

# Medium version
python eval.py --benchmark winogrande:m --split validation

# Small version
python eval.py --benchmark winogrande:s --split validation

# Extra small version
python eval.py --benchmark winogrande:xs --split validation
```

### 7. GSM8K (Grade School Math 8K)

Math word problems requiring multi-step reasoning. Uses generative evaluation with exact match on final numerical answer.

```bash
# Test split (1319 examples)
python eval.py --benchmark gsm8k --split test

# Train split (7473 examples, for few-shot examples)
python eval.py --benchmark gsm8k --split train

# Quick test with limited samples
python eval.py --benchmark gsm8k --split test --max_samples 100
```

**Note**: GSM8K uses **generative evaluation** (generates full solution) unlike other benchmarks which use log probability scoring. This makes it slower but tests actual reasoning ability.

## Common Options

### Checkpoint Selection

```bash
# Use a specific checkpoint
python eval.py --benchmark hellaswag --checkpoint checkpoints/my_model.pt

# Default checkpoint
python eval.py --benchmark hellaswag  # Uses checkpoints/seqcond_torch_80k.pt
```

### Limiting Samples

Useful for quick testing or debugging:

```bash
# Evaluate only first 100 samples
python eval.py --benchmark hellaswag --max_samples 100

# Evaluate only first 50 samples of GPQA
python eval.py --benchmark gpqa:diamond --max_samples 50
```

### Split Selection

```bash
# Validation split (common for development)
python eval.py --benchmark hellaswag --split validation

# Test split (for final evaluation)
python eval.py --benchmark hellaswag --split test

# Train split (rarely used for evaluation)
python eval.py --benchmark mmlu:anatomy --split train
```

## HuggingFace Authentication (for GPQA)

GPQA is a gated dataset requiring HuggingFace authentication:

1. **Get access**: Visit https://huggingface.co/datasets/Idavidrein/gpqa and request access
2. **Create token**: Go to https://huggingface.co/settings/tokens
   - Create a token with "Read access to contents of all public gated repos you can access"
3. **Set token**: The token is already configured in `eval.py` (line 5)

## Example Evaluation Suite

Run a comprehensive evaluation across multiple benchmarks:

```bash
# Quick evaluation suite (with limited samples)
python eval.py --benchmark hellaswag --split validation --max_samples 500
python eval.py --benchmark arc:challenge --split test --max_samples 500
python eval.py --benchmark piqa --split validation --max_samples 500
python eval.py --benchmark winogrande:xl --split validation --max_samples 500
python eval.py --benchmark gsm8k --split test --max_samples 100

# Full evaluation suite (takes longer)
python eval.py --benchmark hellaswag --split validation
python eval.py --benchmark arc:easy --split test
python eval.py --benchmark arc:challenge --split test
python eval.py --benchmark piqa --split validation
python eval.py --benchmark winogrande:xl --split validation
python eval.py --benchmark gpqa:diamond
python eval.py --benchmark gsm8k --split test
python eval.py --benchmark mmlu:anatomy --split test
python eval.py --benchmark mmlu:college_physics --split test
```

## Output Format

All benchmarks provide:
- **Real-time progress**: Updates every 10-100 samples
- **Accuracy**: Percentage and fraction (correct/total)
- **Speed**: Samples per second
- **Time**: Total evaluation time

Example output:
```
Evaluating on 10042 samples...
  100/10042 | Acc: 65.0% | Speed: 4.2 samples/s
  200/10042 | Acc: 63.5% | Speed: 4.3 samples/s
  ...

============================================================
Results:
  Accuracy: 64.23% (6451/10042)
  Time: 2341.2s
  Speed: 4.3 samples/s
============================================================
```

## Notes

- All evaluations use **log probability scoring** for multiple-choice questions
- The model selects the answer with the highest average log probability
- **No fine-tuning** is performed - this is zero-shot evaluation
- Evaluation is performed on GPU with CUDA graphs for optimal speed
