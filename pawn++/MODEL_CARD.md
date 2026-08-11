---
license: mit
language:
- en
tags:
- text-classification
- ai-generated-text-detection
- machine-generated-text
- llm-detection
- pawn
pipeline_tag: text-classification
metrics:
- accuracy
- f1
- precision
- recall
- roc_auc
base_model:
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-1B
datasets:
- yaful/MAGE
model-index:
- name: PAWN++
  results:
  - task:
      type: text-classification
      name: Machine-Generated Text Detection
    dataset:
      name: MAGE
      type: yaful/MAGE
    metrics:
    - type: accuracy
      value: 0.9515
    - type: f1_macro
      value: 0.9515
    - type: roc_auc
      value: 0.9836
    - type: f1
      value: 0.9506
      name: AI F1
    - type: f1
      value: 0.9523
      name: Human F1
---

# PAWN++

[![GitHub](https://img.shields.io/badge/GitHub-automatic--goggles-181717?logo=github)](https://github.com/HSE-Team-142/automatic-goggles/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/HSE-Team-142/automatic-goggles/blob/main/LICENSE)
[![Task](https://img.shields.io/badge/task-AI%20text%20detection-blue)](#)
[![Accuracy](https://img.shields.io/badge/accuracy-95.15%25-success)](#evaluation)
[![ROC AUC](https://img.shields.io/badge/ROC%20AUC-0.984-success)](#evaluation)
[![Macro F1](https://img.shields.io/badge/macro%20F1-0.951-success)](#evaluation)

📦 **Repository:** [HSE-Team-142/automatic-goggles](https://github.com/HSE-Team-142/automatic-goggles/)

**PAWN++** is a detector for identifying machine-generated (AI) text. It extends the
[PAWN](https://www.sciencedirect.com/science/article/pii/S156625352500538X) architecture, which
predicts authorship from the per-token hidden states and probability metrics of a *frozen* language
model. PAWN++ adds an optional second frozen language model, cross-model metrics (including a
Binoculars-style cross-perplexity score), second-model token metrics, hidden-state fusion, and
aggregated sequence-level features that modulate the representation through FiLM.

This card describes the best-performing configuration
(`mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full`, checkpoint `checkpoint-39884`).

## Model Description

PAWN++ does **not** fine-tune the backbone LLMs. The two language models are frozen and used only as
feature extractors; the only trained parameters are three lightweight MLP heads and a FiLM
conditioning layer.

- **Model type:** Frozen-LLM feature extractor + gated MLP classifier (binary)
- **Task:** Binary classification — `human` (0) vs. `ai` (1)
- **Primary model (frozen):** `meta-llama/Llama-3.2-1B-Instruct`
- **Second model (frozen):** `meta-llama/Llama-3.2-1B`
- **Language:** English
- **Max sequence length:** 512 tokens
- **License:** MIT

### Architecture

For each input the feature extractor runs both frozen LLMs and produces:

1. **Per-token metrics** for each model — `entropy`, `max_log_probs`, `next_token_log_probs`,
   `rank`, `top_p` — plus the **cross-perplexity (xppl)** between the two models.
2. **Hidden states** from both models, fused across layers with `uniform` fusion.
3. **Aggregated sequence-level features** — per model: `energy`, `mean`, `std`, `var`, `skew`,
   `kurtosis`, `mean_diff`, `std_diff`, `var_2nd`, `entropy_2nd`, `autocorr_2nd`; and **cross-model**:
   `cov`, `corr`, `cos_sim`, `binoculars_score`.

Three MLP heads process these signals:

- **`metrics_nn`** maps the per-token metric vector to a 256-dim feature space.
- **`gate_nn`** takes the concatenated current/next hidden states of both models plus a positional
  scalar and produces 256 gate logits per token; a softmax over the sequence axis yields an
  attention-style weighting that aggregates the token metric features into a single vector.
- The aggregated vector is modulated by a **FiLM** layer (`gamma`, `beta`) conditioned on the
  normalized sequence-level aggregate features.
- **`aggregate_nn`** maps the result to a single logit.

The output is a single logit; `sigmoid(logit)` is the probability of the `human` class and the
prediction is `ai` when `logit >= 0`.

| Hyperparameter | Value |
|---|---|
| `metric_features` | 256 |
| `gates` | 256 |
| `mlp_hidden_features` | 256 |
| `mlp_hidden_layers` | 3 |
| `mlp_dropout` | 0.0 |
| `token_dropout` | 0.15 |
| `residual` | true |
| `hidden_state_fusion` | uniform |

## Intended Use

- **Primary use:** Research on machine-generated-text detection and AI-text classification of English
  passages.
- **Out of scope:** High-stakes decisions (academic misconduct, hiring, moderation) without human
  review; non-English text; short texts; and detecting generators or domains far from the training
  distribution. As with all detectors, predictions should be treated as a signal, not proof.

## Training Data

Trained and evaluated on the **MAGE** benchmark for machine-generated text detection, which spans
multiple domains and many generator models, framed as a binary human-vs-AI task.

## Training Procedure

- Backbones frozen; only the MLP heads and FiLM layer are trained.
- **Objective:** Binary cross-entropy with `label_smoothing = 0.2` and `pos_weight = 0.413`.
- **Optimizer:** AdamW, `learning_rate = 1e-3`, `weight_decay = 1e-2`, `max_grad_norm = 1.0`.
- **Schedule:** up to 5 epochs (`max_steps = 49855`), batch size 32, early stopping (patience 5),
  seed 42.
- **Model selection:** best checkpoint by validation AUROC (`checkpoint-39884`, epoch 4, validation
  AUROC ≈ **0.9933**).

## Evaluation

Results on the MAGE test set:

| Metric | Value |
|---|---|
| Accuracy | 0.9515 |
| Macro F1 | 0.9515 |
| ROC AUC | 0.9836 |
| AI — Precision | 0.9710 |
| AI — Recall | 0.9311 |
| AI — F1 | 0.9506 |
| Human — Precision | 0.9334 |
| Human — Recall | 0.9720 |
| Human — F1 | 0.9523 |
| Test loss | 0.2456 |

**Runtime (test split):** 1619.5 s, 37.5 samples/s, 1.173 steps/s.

> The model is slightly more precise on AI text (fewer false AI flags) and has higher recall on human
> text, i.e. it is conservative about labeling text as AI-generated.

## How to Use

Inference is provided through `inference.py`, which loads the frozen backbones plus the trained heads
from a checkpoint and a training YAML config:

```bash
uv run pawn++/inference.py \
  --config pawn++/experiments/MAGE/configs/pawn/two_models/mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full.yaml \
  --checkpoint pawn++/checkpoint-39884/pytorch_model.bin \
  --text "Your text to classify here."
```

```python
from inference import load_model, predict

model, device = load_model(
    config_path="pawn++/experiments/MAGE/configs/pawn/two_models/"
                "mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full.yaml",
    checkpoint_path="pawn++/checkpoint-39884/pytorch_model.bin",
)
results = predict(model, ["Your text to classify here."], device)
# each result: {"label": "human"|"ai", "prediction": 0|1, "prob_human": float, "logit": float}
```

> **Note:** The Llama-3.2 backbones are gated on the Hugging Face Hub. Set `HF_TOKEN` in a `.env` file
> to download them. A GPU is recommended; the code falls back to MPS or CPU automatically.

## Limitations and Bias

- English-only; performance on other languages is not evaluated and expected to degrade.
- Detection quality depends on the generators and domains seen during training (MAGE); novel models,
  prompting styles, paraphrasing or adversarial edits can reduce accuracy.
- Depends on two frozen Llama-3.2-1B backbones, which carry their own data biases.
- Reported metrics reflect the MAGE test distribution and may not transfer out of distribution; see
  the OOD evaluation utilities in the repository.

## Citation

PAWN++ builds on the PAWN detector:

> PAWN: Perplexity-Aware Watermark-free News (machine-generated text detection).
> https://www.sciencedirect.com/science/article/pii/S156625352500538X
