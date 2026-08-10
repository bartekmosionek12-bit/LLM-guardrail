# 🛡️ Aegis AI Guardrail Classifier

A highly optimized, lightweight, and trainable AI Guardrail designed to protect Large Language Models (LLMs) from Prompt Injections, Jailbreaks, and Toxic inputs. 

This module fine-tunes a `distilbert-base-uncased` model using HuggingFace Transformers. It has been specifically optimized to train extremely fast on consumer-grade GPUs with 8GB VRAM (e.g., NVIDIA RTX 3070) utilizing FP16 mixed-precision training.

## ✨ Features

- **Multi-class Classification**: Accurately differentiates between `Benign` (safe), `Prompt Injection` (attacks), and `Toxic/Harmful` inputs.
- **VRAM Optimized**: Uses `batch_size=16` and `fp16=True` to train quickly without out-of-memory (OOM) errors on 8GB GPUs.
- **Production-Ready UI**: Includes a Gradio-based web interface for instant, real-time testing of the trained model.
- **HuggingFace Integration**: Built-in script to download and parse the massive `JailbreakV-28K` dataset for high-quality threat intelligence.
- **Active Learning (RLHF-lite)**: The UI features custom correction buttons. If the model misclassifies a query, you can manually flag the correct label, which instantly appends the data back to the training dataset for continuous improvement.

---

## 📂 Project Structure

- `dataset_builder.py` — Connects to the HuggingFace Hub, downloads the `JailbreakV-28K` dataset, balances it with synthetic benign queries, and generates a ready-to-use CSV.
- `train.py` — The core fine-tuning script. Loads the dataset, configures the DistilBERT architecture, and trains the model on the GPU. Saves the final weights to `./model_weights`.
- `app.py` — A Gradio web application for inference. Loads the `.pth` weights into memory and exposes a UI to test the model against custom prompts.

---

## 🚀 Getting Started

### 1. Requirements

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
# It is recommended to run this inside a virtual environment (venv or conda)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers datasets pandas gradio accelerate
```

### 2. Build the Dataset

Run the builder script to download the datasets and prepare the `guardrail_dataset.csv`. This dataset will contain over 30,000 records.

```bash
python dataset_builder.py
```

### 3. Train the Model

Start the training process. On an RTX 3070, this typically takes around 10-15 minutes for 5 epochs. 

```bash
python train.py
```
*Note: Make sure your terminal outputs `Trening odbędzie się na GPU! 🔥` at the start to ensure CUDA is being utilized.*

### 4. Test in the UI

Once training is complete, launch the evaluation server:

```bash
python app.py
```
Open the provided URL (e.g., `http://127.0.0.1:7860`) in your browser to test the Guardrail in real-time. Use the correction buttons in the UI to teach the model its mistakes and grow your dataset!

---

## 📊 Classification Labels

The model outputs one of three confidence-scored labels:
- **`0` (✅ Benign)**: Safe, standard interactions. Edge cases (like talking *about* prompt injection) are also routed here.
- **`1` (⚠️ Prompt Injection)**: System override attempts, DAN-style jailbreaks, and instructions evasion.
- **`2` (⛔ Toxic / Harmful)**: Generation of illegal, unethical, or discriminatory content.

---
*Created as part of the Aegis Secure LLM Gateway.*
