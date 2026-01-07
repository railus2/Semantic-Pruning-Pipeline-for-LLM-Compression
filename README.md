# Semantic Structured Pruning Pipeline for LLM Compression

## Overview
This project presents an end-to-end pipeline for Large Language Model (LLM) compression based on semantic structured pruning, followed by a lightweight recovery tuning stage. Unlike common compression approaches such as quantization or distillation, this work modifies the model architecture itself by removing structurally unimportant components (attention heads and MLP neurons) according to their semantic contribution. The entire pipeline is designed to be fully reproducible on Kaggle GPUs, with clear ablation studies and quantitative trade-off analysis.

## Models and Stages

### 1. Semantic Importance Analysis
Discover and rank components (heads, neurons) based on their semantic importance metrics.

### 2. Structured Semantic Pruning
Perform parameter reduction by pruning structurally unimportant elements.

### 3. Recovery Tuning
Lightweight LoRA adapters fine-tune pruned models to recover quality.

### 4. Evaluation
Tradeoff analysis (perplexity, latency, size).

## Repository Files
- **notebooks/** for pipeline implementation
- **models/** saved pruned/recovered files <br>

Reproducible on Kaggle with environment setup. All components retrainable.