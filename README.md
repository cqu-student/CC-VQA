# CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering
[![arXiv](https://img.shields.io/badge/arXiv-2510.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2510.XXXXX)
[![CVPR 2026](https://img.shields.io/badge/CVPR%202026-Paper-blue)](https://cvpr.thecvf.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-orange?logo=huggingface&logoColor=white)](https://huggingface.co/)

This repository provides the official implementation for **CC-VQA**, a novel **training-free**, conflict- and correlation-aware method for Knowledge-Based Visual Question Answering (KB-VQA). CC-VQA addresses the critical challenge of knowledge conflicts between parametric knowledge in VLMs and retrieved external information.

## 🪵 TODO List

- [ ] Release core implementation
- [ ] Complete README documentation
- [ ] Release evaluation scripts
- [ ] Add detailed Quick Start guide

## 🔥 What's New

- **(2026.02.20)** Project initialized.
- **(2026.02.20)** 📄 Paper accepted by CVPR 2026.

# 🧠 CC-VQA: A Training-Free Conflict-Aware Framework for KB-VQA

> Official implementation of CC-VQA.

![Method Overview](img/fig_overview_v3.jpg)
*CC-VQA achieves state-of-the-art results on KB-VQA benchmarks by effectively mitigating knowledge conflicts.*

---

## 📌 Abstract

Knowledge-based visual question answering (KB-VQA) demonstrates significant potential for handling knowledge-intensive tasks. However, conflicts arise between static parametric knowledge in vision language models (VLMs) and dynamically retrieved information due to the static model knowledge from pre-training. Current mitigation methods often neglect visual information or suffer from redundant contexts.

We propose **CC-VQA**, a novel training-free method comprising two core components:

- 👁️ **Vision-Centric Contextual Conflict Reasoning**: Performs visual-semantic conflict analysis across internal and external knowledge contexts.
- 📉 **Correlation-Guided Encoding and Decoding**: Features positional encoding compression for low-correlation statements and adaptive decoding using correlation-weighted conflict scoring.

Extensive evaluations on E-VQA, InfoSeek, and OK-VQA benchmarks demonstrate that **CC-VQA achieves state-of-the-art performance**, yielding absolute accuracy improvements of **3.3% to 6.4%** compared to existing methods.

---

## 🏗️ Architecture

Our framework consists of two main innovative modules designed to handle knowledge conflicts without fine-tuning:

1. **Vision-Centric Contextual Conflict Reasoning**  
   Analyzes conflicts between the model's internal knowledge and retrieved external knowledge, with a specific focus on visual cues to ensure generated answers are grounded in the image content.

2. **Correlation-Guided Encoding and Decoding**  
   - **Encoding**: Compresses low-correlation statements using positional encoding to reduce noise.
   - **Decoding**: Utilizes correlation-weighted conflict scoring to adaptively generate the final answer, favoring high-confidence, non-conflicting information.

---

## 📊 Results

### Performance on OK-VQA

Our method achieves state-of-the-art performance of **78.8%** on OK-VQA, surpassing both fine-tuned and training-free baselines.

| Method | Model | Gen. FT | Accuracy |
|--------|-------|---------|----------|
| Qwen2.5-VL-7B | - | - | 72.4 |
| Wiki-PRF-7B | Qwen2.5-VL-7B | ✅ | 77.8 |
| MMKB-RAG | LLaMA-3.1-8B | ❌ | 65.4 |
| KU-RAG | LLaVA-Next-7B | ❌ | 73.1 |
| **CC-VQA (Ours)** | **Qwen2.5-VL-7B** | **❌** | **78.8** |

### Main Results on E-VQA and InfoSeek

Standard retrieval augmentation boosts Qwen2.5-VL-7B performance significantly. **CC-VQA** achieves further improvements demonstrating effective knowledge conflict resolution.

| Method | Model | Gen. FT | Retriever | E-VQA (Single-Hop) | E-VQA (All) | InfoSeek (Unseen-Q) | InfoSeek (Unseen-E) | InfoSeek (All) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| *Zero-shot MLLMs* | | | | | | | | |
| BLIP-2 | Flan-T5 XL | - | - | 12.6 | 12.4 | 12.7 | 12.3 | 12.5 |
| InstructBLIP | Flan-T5 XL | - | - | 11.9 | 12.0 | 8.9 | 7.4 | 8.1 |
| LLaVA-v1.5 | Vicuna-7B | - | - | 16.3 | 16.9 | 9.6 | 9.4 | 9.5 |
| GPT-4V | - | - | - | 26.9 | 28.1 | 15.0 | 14.3 | 14.6 |
| Qwen2.5-VL-7B | - | - | - | 21.7 | 20.3 | 22.8 | 24.1 | 23.7 |
| *Retrieval-Augmented* | | | | | | | | |
| DPR V+T | Multi-passage BERT | - | CLIP ViT-B/32 | 29.1 | - | - | - | 12.4 |
| RORA-VLM | Vicuna-7B | ✅ | CLIP+Google | - | 20.3 | 25.1 | 27.3 | - |
| EchoSight | Mistral/LLaMA-3 | ❌ | EVA-CLIP-8B | 26.4 | 24.9 | 30.0 | 30.7 | 30.4 |
| Wiki-LLaVA | Vicuna-7B | ✅ | Contriever | 17.7 | 20.3 | 30.1 | 27.8 | 28.9 |
| ReflectiVA | LLaMA-3.1-8B | ✅ | EVA-CLIP-8B | 28.0 | 29.2 | 40.4 | 39.8 | 40.1 |
| MMKB-RAG | Qwen2-7B | ❌ | EVA-CLIP-8B | 39.7 | 35.9 | 36.4 | 36.3 | 36.4 |
| Wiki-PRF | Qwen2.5-VL-8B | ✅ | EVA-CLIP-8B | 37.1 | 36.0 | 43.3 | 42.7 | 42.8 |
| Qwen2.5-VL-7B (Base) | Qwen2.5-VL-7B | ❌ | EVA-CLIP-8B | 36.7 | 31.2 | 41.9 | 41.3 | 41.8 |
| **CC-VQA (Ours)** | **Qwen2.5-VL-7B** | **❌** | **EVA-CLIP-8B** | **41.4** | **36.1** | **44.7** | **46.1** | **45.1** |

- **E-VQA**: Improved by **+4.7%** over strong baselines.
- **InfoSeek**: Improved by **+3.3%** over strong baselines.
- **Oracle Analysis**: Achieves **66.5%** accuracy on InfoSeek when provided with ground-truth articles, showing superior information utilization.

---

## 🚀 Get Started

*(Code release coming soon)*

```bash
git clone https://github.com/cqu-student/CC-VQA.git
cd CC-VQA
# pip install -r requirements.txt
```
