# CC-VQA-Contextual-Conflict-aware-Visual-Question-Answering
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
