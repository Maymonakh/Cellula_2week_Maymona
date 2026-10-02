# Cellula Internship — 2 Week Tasks

This repository contains my solutions for the tasks completed during my 2-week internship at **Cellula Technologies**.

## Task 0 — Quantization

Research and practical examples about **quantization** for large language models such as BERT and LLaMA.

The task covers:

* Why large models require a lot of memory
* 8-bit and 4-bit quantization
* Using quantization with `bitsandbytes`
* Memory usage comparison
* Advantages and limitations of quantization

📁 [Task 0](./Task0)

---

## Task 1 — Toxic Content Classification

The goal was to classify text into **9 toxic content categories** using different deep learning approaches.

The dataset contains **3000 samples** with:

* `query`
* `image descriptions`
* `Toxic Category`

The two text fields were combined and used as the input for classification.

### LSTM

A Keras/TensorFlow LSTM model was trained using:

```text
Text
 ↓
Tokenization
 ↓
Embedding
 ↓
LSTM
 ↓
Dense Layers
 ↓
Classification
```

**Results:**

| Metric      |      Score |
| ----------- | ---------: |
| Weighted F1 | **0.9544** |
| Macro F1    | **0.9515** |

🚀 **Live App:** https://lstm-app.streamlit.app/

### DistilBERT + LoRA

A `distilbert-base-uncased` model was fine-tuned using **LoRA**.

Only about **1.1% of the model parameters** were trainable.

**Results:**

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **0.9183** |
| Weighted F1 | **0.8967** |
| Macro F1    | **0.8277** |

🚀 **Live App:** https://distilbert-app.streamlit.app/

### Image Classification

For image input, **BLIP** was used to generate an image caption. The generated caption was then passed to the text classification model.

### Database

A simple **SQLite** database was used to save classification results.

### Technologies

* Python
* TensorFlow / Keras
* PyTorch
* Hugging Face Transformers
* LoRA
* BLIP
* Scikit-learn
* SQLite
* Streamlit

## Run the Application Locally

Go to either Task 1 folder:

```bash
cd Task1_LSTM
```

or:

```bash
cd Task1_DistilBERT
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application allows you to:

* Classify text
* Upload an image and classify it
* View classification history from the SQLite database

