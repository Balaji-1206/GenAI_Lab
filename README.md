# 🧠 Generative AI Laboratory (`GenAI_Lab`)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%26%20Diffusers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-00599C?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Deployment-Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)

A comprehensive, hands-on repository containing **12 foundational to advanced Generative AI experiments**. This laboratory covers modern Generative AI paradigms including Large Language Models (LLMs), prompt engineering, multi-turn conversational agents, task-specific NLP pipelines, Retrieval-Augmented Generation (RAG) with vector databases, code generation, latent diffusion models, vision-language multimodal models, supervised fine-tuning, automated multimedia generation, and cloud-ready web deployment with quantitative evaluation metrics.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Lab Experiments Index](#-lab-experiments-index)
- [Detailed Experiment Breakdown](#-detailed-experiment-breakdown)
  - [1. Text Generation Using Pre-Trained Foundation Models](#1-text-generation-using-pre-trained-foundation-models)
  - [2. Prompt Engineering Techniques](#2-prompt-engineering-techniques)
  - [3. Conversational AI Chatbot](#3-conversational-ai-chatbot)
  - [4. Text Summarization & Question-Answering](#4-text-summarization--question-answering)
  - [5. Sentiment Analysis & Zero-Shot Document Classification](#5-sentiment-analysis--zero-shot-document-classification)
  - [6. Retrieval-Augmented Generation (RAG) with Vector Databases](#6-retrieval-augmented-generation-rag-with-vector-databases)
  - [7. AI-Powered Code Generation & Debugging Assistant](#7-ai-powered-code-generation--debugging-assistant)
  - [8. Image Generation Using Diffusion Models](#8-image-generation-using-diffusion-models)
  - [9. Multimodal Vision-Language Application](#9-multimodal-vision-language-application)
  - [10. Domain-Specific Fine-Tuning of a Pre-Trained LLM](#10-domain-specific-fine-tuning-of-a-pre-trained-llm)
  - [11. AI-Based Multimedia Content Generation System](#11-ai-based-multimedia-content-generation-system)
  - [12. Deployment & Evaluation of Generative AI Applications](#12-deployment--evaluation-of-generative-ai-applications)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Technology Stack](#-technology-stack)
- [Author](#-author)

---

## 🌟 Overview

This repository serves as a complete practical curriculum for Generative AI, featuring clean implementations, sample input/output logs, and generated visual artifacts across 12 structured experiments:

- **Language Modeling & Prompting:** Autoregressive generation, sampling techniques (top-k, top-p, temperature), Zero-shot, Few-shot, and Chain-of-Thought (CoT) prompting.
- **Dialogue & Task-Oriented AI:** State-managed conversational agents (`DialoGPT`), abstractive summarization (`BART`), extractive reading comprehension (`DistilBERT-SQuAD`), and Zero-shot classification (`BART-MNLI`).
- **Knowledge Augmentation:** Dense embeddings (`SentenceTransformers`), vector similarity search (`FAISS`), and contextual synthesis (`FLAN-T5`).
- **Code & Multimodal Intelligence:** Code synthesis and repair (`CodeGen`), latent diffusion image generation (`Stable Diffusion v1.5`), and visual question answering & captioning (`BLIP`).
- **Fine-Tuning & Productionization:** Transfer learning with Hugging Face `Trainer`, cross-modal multimedia synthesis (`FLAN-T5` + `Stable Diffusion` + `gTTS`), and web interfaces with `Gradio` benchmarked with `ROUGE` evaluation scores.

---

## 📚 Lab Experiments Index

| # | Experiment Title | Model / Architecture | Key Focus & Techniques | Folder |
|:--|:-----------------|:---------------------|:-----------------------|:-------|
| **01** | [Text Generation](#1-text-generation-using-pre-trained-foundation-models) | `gpt2` | Autoregressive sampling, Temperature, Top-$k$, Top-$p$ | [`exp_1/`](./exp_1/) |
| **02** | [Prompt Engineering](#2-prompt-engineering-techniques) | `gpt2` | Zero-Shot, Few-Shot, Chain-of-Thought (CoT) | [`exp_2/`](./exp_2/) |
| **03** | [Conversational Chatbot](#3-conversational-ai-chatbot) | `microsoft/DialoGPT-medium` | Multi-turn dialogue, context token concatenation | [`exp_3/`](./exp_3/) |
| **04** | [Summarization & QA](#4-text-summarization--question-answering) | `facebook/bart-large-cnn`, `distilbert-base-cased-distilled-squad` | Abstractive summarization & Extractive span QA | [`exp_4/`](./exp_4/) |
| **05** | [Sentiment & Classification](#5-sentiment-analysis--zero-shot-document-classification) | `distilbert-base-uncased-finetuned-sst-2-english`, `facebook/bart-large-mnli` | Sentiment scoring & Zero-Shot NLI classification | [`exp_5/`](./exp_5/) |
| **06** | [RAG with Vector DB](#6-retrieval-augmented-generation-rag-with-vector-databases) | `all-MiniLM-L6-v2`, `FAISS`, `google/flan-t5-base` | Dense embeddings, vector similarity indexing, grounded generation | [`exp_6/`](./exp_6/) |
| **07** | [Code Generation & Debugging](#7-ai-powered-code-generation--debugging-assistant) | `Salesforce/codegen-350M-mono` | Natural language to code, automated bug fixing | [`exp_7/`](./exp_7/) |
| **08** | [Diffusion Image Generation](#8-image-generation-using-diffusion-models) | `runwayml/stable-diffusion-v1-5` | Latent diffusion, prompt guidance scale, inference steps | [`exp_8/`](./exp_8/) |
| **09** | [Multimodal Text & Image](#9-multimodal-vision-language-application) | `Salesforce/blip-image-captioning-base`, `Salesforce/blip-vqa-base` | Image Captioning & Visual Question Answering (VQA) | [`exp_9/`](./exp_9/) |
| **10** | [Domain Fine-Tuning](#10-domain-specific-fine-tuning-of-a-pre-trained-llm) | `distilbert-base-uncased`, `IMDB` | Supervised Fine-Tuning, Hugging Face `Trainer`, Accuracy | [`exp_10/`](./exp_10/) |
| **11** | [Multimedia Content Generation](#11-ai-based-multimedia-content-generation-system) | `FLAN-T5` + `Stable Diffusion` + `gTTS` | Multi-modal pipeline: Text synthesis + Image art + Speech audio | [`exp_11/`](./exp_11/) |
| **12** | [Deployment & Evaluation](#12-deployment--evaluation-of-generative-ai-applications) | `BART-CNN`, `Gradio`, `evaluate (ROUGE)` | Web deployment, cloud sharing, ROUGE-1/2/L evaluation metrics | [`exp_12/`](./exp_12/) |

---

## 🔬 Detailed Experiment Breakdown

### 1. Text Generation Using Pre-Trained Foundation Models
- **File:** [`Ex1_Text_Generation.py`](./Ex1_Text_Generation.py) / [`exp_1/code.py`](./exp_1/code.py)
- **Model:** `gpt2` (OpenAI)
- **Concepts:** Autoregressive sequence modeling, stochastic sampling, controlling generation creativity and diversity via `temperature=0.8`, `top_k=50`, and nucleus sampling `top_p=0.95`.
- **Output:** Multi-sequence continuation from prompt *"Artificial Intelligence will transform the future of..."*.

### 2. Prompt Engineering Techniques
- **File:** [`Ex2_Prompt_Engineering.py`](./Ex2_Prompt_Engineering.py) / [`exp_2/code.py`](./exp_2/code.py)
- **Model:** `gpt2`
- **Concepts:** Exploring prompt structuring strategies without parameter updates:
  - **Zero-Shot Prompting:** Direct instruction for sentiment classification.
  - **Few-Shot Prompting:** Providing exemplars to guide output formatting and label mapping.
  - **Chain-of-Thought (CoT) Prompting:** Guiding multi-step arithmetic reasoning (*"Let's think step by step"*).

### 3. Conversational AI Chatbot
- **File:** [`Ex3_Conversational_Chatbot.py`](./Ex3_Conversational_Chatbot.py) / [`exp_3/code.py`](./exp_3/code.py)
- **Model:** `microsoft/DialoGPT-medium`
- **Concepts:** Multi-turn conversational modeling, maintaining dialogue history by appending user and system response token IDs across turns with `torch.cat` and special end-of-string tokens (`eos_token`).

### 4. Text Summarization & Question-Answering
- **File:** [`Ex4_Summarization_QA.py`](./Ex4_Summarization_QA.py) / [`exp_4/code.py`](./exp_4/code.py)
- **Models:** `facebook/bart-large-cnn` (Summarization) & `distilbert-base-cased-distilled-squad` (QA)
- **Concepts:**
  - **Abstractive Summarization:** Condensing long articles into concise overviews using sequence-to-sequence bidirectional autoregressive transformers.
  - **Extractive QA:** Finding answer spans and calculating confidence scores over passage contexts.

### 5. Sentiment Analysis & Zero-Shot Document Classification
- **File:** [`Ex5_Sentiment_Classification.py`](./Ex5_Sentiment_Classification.py) / [`exp_5/code.py`](./exp_5/code.py)
- **Models:** `distilbert-base-uncased-finetuned-sst-2-english` & `facebook/bart-large-mnli`
- **Concepts:**
  - **Sentiment Polarity Scoring:** Classifying user reviews into positive/negative classes with confidence probabilities.
  - **Zero-Shot NLI Classification:** Categorizing unlabelled text across arbitrary candidate domains (Politics, Economy, Sports, Technology) using natural language inference hypothesis formulation.

### 6. Retrieval-Augmented Generation (RAG) with Vector Databases
- **File:** [`Ex6_RAG_VectorDB.py`](./Ex6_RAG_VectorDB.py) / [`exp_6/code.py`](./exp_6/code.py)
- **Models & Tools:** `sentence-transformers/all-MiniLM-L6-v2`, `faiss` (Facebook AI Similarity Search), `google/flan-t5-base`
- **Concepts:** Mitigating LLM hallucinations through retrieval grounding:
  1. Chunking and dense vector embedding of a custom knowledge base.
  2. Building an exact L2 distance FAISS vector index (`IndexFlatL2`).
  3. Query vector similarity search to fetch top-$k$ relevant chunks.
  4. Augmented prompt construction and answer synthesis via instruction-tuned FLAN-T5.

### 7. AI-Powered Code Generation & Debugging Assistant
- **File:** [`Ex7_Code_Generation_Debugging.py`](./Ex7_Code_Generation_Debugging.py) / [`exp_7/code.py`](./exp_7/code.py)
- **Model:** `Salesforce/codegen-350M-mono`
- **Concepts:**
  - **Code Synthesis:** Generating functional Python implementations (e.g., `is_prime(n)`) from docstrings and function signatures.
  - **Automated Code Debugging:** Detecting algorithmic errors (e.g., incorrect factorial base initialization) and providing corrected code completions.

### 8. Image Generation Using Diffusion Models
- **File:** [`Ex8_Image_Generation_Diffusion.py`](./Ex8_Image_Generation_Diffusion.py) / [`exp_8/code.py`](./exp_8/code.py)
- **Model:** `runwayml/stable-diffusion-v1-5`
- **Concepts:** Latent diffusion architectures, text-to-image synthesis, classifier-free guidance (`guidance_scale=7.5`), and iterative noise scheduling across inference steps with FP16 CUDA acceleration.

### 9. Multimodal Vision-Language Application
- **File:** [`Ex9_Multimodal_Text_Image.py`](./Ex9_Multimodal_Text_Image.py) / [`exp_9/code.py`](./exp_9/code.py)
- **Models:** `Salesforce/blip-image-captioning-base` & `Salesforce/blip-vqa-base` (BLIP)
- **Concepts:** Cross-modal interaction between visual encoders and text decoders:
  - **Image Captioning:** Generating semantic descriptions of open-domain images.
  - **Visual Question Answering (VQA):** Answering natural language questions conditioned on visual features.

### 10. Domain-Specific Fine-Tuning of a Pre-Trained LLM
- **File:** [`Ex10_Fine_Tuning.py`](./Ex10_Fine_Tuning.py) / [`exp_10/code.py`](./exp_10/code.py)
- **Model & Dataset:** `distilbert-base-uncased` fine-tuned on the `IMDB` sentiment dataset
- **Concepts:** Transfer learning with domain adaptation, dynamic tokenization with truncation & padding, Hugging Face `TrainingArguments`, `Trainer` loop orchestration, evaluation metric calculation (`accuracy_score`), and saving fine-tuned weights for downstream inference.

### 11. AI-Based Multimedia Content Generation System
- **File:** [`Ex11_Content_Generation_Multimedia.py`](./Ex11_Content_Generation_Multimedia.py) / [`exp_11/code.py`](./exp_11/code.py)
- **Models & Tools:** `google/flan-t5-base`, `runwayml/stable-diffusion-v1-5`, `gTTS` (Google Text-to-Speech)
- **Concepts:** Building an automated tri-modal content creation engine from a single topic prompt:
  1. **Text:** Informative narrative generation via FLAN-T5.
  2. **Visuals:** Thematic digital art generation via Stable Diffusion.
  3. **Audio:** High-quality voiceover narration generated via Text-to-Speech (TTS).

### 12. Deployment & Evaluation of Generative AI Applications
- **File:** [`Ex12_Deployment_Evaluation.py`](./Ex12_Deployment_Evaluation.py) / [`exp_12/code.py`](./exp_12/code.py)
- **Frameworks:** `Gradio`, `facebook/bart-large-cnn`, Hugging Face `evaluate`
- **Concepts:**
  - **Application Deployment:** Packaging the generative model into an interactive Web UI with instant public cloud URL sharing (`share=True`).
  - **Quantitative Metric Evaluation:** Evaluating generated text against human references using n-gram overlap metrics (**ROUGE-1**, **ROUGE-2**, **ROUGE-L**, **ROUGE-Lsum**).

---

## 📁 Repository Structure

```plaintext
GenAI_Lab/
├── Ex1_Text_Generation.py                  # Standalone Script - Exp 1
├── Ex2_Prompt_Engineering.py               # Standalone Script - Exp 2
├── Ex3_Conversational_Chatbot.py           # Standalone Script - Exp 3
├── Ex4_Summarization_QA.py                 # Standalone Script - Exp 4
├── Ex5_Sentiment_Classification.py         # Standalone Script - Exp 5
├── Ex6_RAG_VectorDB.py                     # Standalone Script - Exp 6
├── Ex7_Code_Generation_Debugging.py        # Standalone Script - Exp 7
├── Ex8_Image_Generation_Diffusion.py       # Standalone Script - Exp 8
├── Ex9_Multimodal_Text_Image.py            # Standalone Script - Exp 9
├── Ex10_Fine_Tuning.py                     # Standalone Script - Exp 10
├── Ex11_Content_Generation_Multimedia.py   # Standalone Script - Exp 11
├── Ex12_Deployment_Evaluation.py           # Standalone Script - Exp 12
├── README.md                               # Comprehensive Project Documentation
│
├── exp_1/                                  # Text Generation Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_2/                                  # Prompt Engineering Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_3/                                  # Chatbot Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_4/                                  # Summarization & QA Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_5/                                  # Sentiment & Classification Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_6/                                  # RAG & Vector DB Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_7/                                  # Code Generation Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_8/                                  # Diffusion Image Generation Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_9/                                  # Multimodal Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_10/                                 # Fine-Tuning Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
├── exp_11/                                 # Multimedia Content Generation Artifacts
│   ├── code.py
│   ├── output.png
│   └── output.txt
└── exp_12/                                 # Deployment & Evaluation Artifacts
    ├── code.py
    ├── output.png
    └── output.txt
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.9+** installed.
- **NVIDIA GPU with CUDA support** is highly recommended for running Diffusion models (Exp 8, 11), Fine-Tuning (Exp 10), and Multimodal models (Exp 9).

### 1. Clone the Repository
```bash
git clone https://github.com/Balaji-1206/GenAI_Lab.git
cd GenAI_Lab
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries with a single command:

```bash
pip install torch torchvision torchaudio \
            transformers \
            diffusers \
            accelerate \
            sentence-transformers \
            faiss-cpu \
            datasets \
            evaluate \
            rouge-score \
            gradio \
            gTTS \
            pillow \
            scikit-learn \
            requests
```

> **Note for GPU Users:** If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch and `faiss-gpu`:
> ```bash
> pip install faiss-gpu
> # Visit https://pytorch.org/get-started/locally/ for the appropriate CUDA wheel command
> ```

---

## 🚀 How to Run

You can execute either the root standalone scripts or the experiment directory scripts.

### Running Individual Experiments
```bash
# Experiment 1: Text Generation
python Ex1_Text_Generation.py

# Experiment 2: Prompt Engineering
python Ex2_Prompt_Engineering.py

# Experiment 3: Conversational Chatbot (Interactive Console)
python Ex3_Conversational_Chatbot.py

# Experiment 6: Retrieval-Augmented Generation (RAG)
python Ex6_RAG_VectorDB.py

# Experiment 8: Stable Diffusion Image Synthesis (requires GPU)
python Ex8_Image_Generation_Diffusion.py

# Experiment 10: Fine-Tuning DistilBERT on IMDB
python Ex10_Fine_Tuning.py

# Experiment 11: Multimedia Content Generation Pipeline
python Ex11_Content_Generation_Multimedia.py

# Experiment 12: Gradio Web UI & Evaluation
python Ex12_Deployment_Evaluation.py
```

Alternatively, navigate into any experiment folder:
```bash
cd exp_6
python code.py
```

---

## 🛠️ Technology Stack

| Category | Frameworks / Libraries |
|:---|:---|
| **Core DL & Runtime** | PyTorch, NumPy, Accelerate |
| **NLP & LLM Architectures** | Hugging Face `transformers`, Tokenizers |
| **Vector DB & Search** | FAISS (`IndexFlatL2`), `sentence-transformers` |
| **Generative Vision** | Hugging Face `diffusers`, Pillow (`PIL`) |
| **Multimodal & Speech** | BLIP (Salesforce), `gTTS` (Google Text-to-Speech) |
| **Datasets & Evaluation** | Hugging Face `datasets`, `evaluate`, `rouge-score`, Scikit-learn |
| **Deployment & UI** | Gradio |

---

## 👤 Author

Developed and maintained by **[Balaji](https://github.com/Balaji-1206)**.

If you find this repository helpful, consider starring ⭐ the repository!