# Voz-Clara: AI-Powered Visual Assistance for the Visually Impaired

**Voz-Clara** is an edge-deployed Visual Question Answering (VQA) system built to assist **visually impaired individuals** in navigating and understanding their environment through spoken interactions. The project leverages the power of **generative vision-language models** and is optimized for real-time use on **Jetson Orin Nano**.

## 🚀 Project Overview

Voz-Clara enables users to **ask questions about their surroundings**, captured via camera, and **receive spoken answers**. Designed to be **fast, accessible, and edge-deployable**, it brings VQA technology to real-world assistive use cases.

---

## 🧠 Core Model

- **Model**: [`PaLI-Gemma 2`](https://ai.googleblog.com/2024/03/pali-gemma-multimodal-models.html)  
- **Dataset**: [VizWiz VQA Dataset](https://vizwiz.org/)  
- **Optimizations**:
  - Fine-tuned for real-world image-question answering
  - Quantized for deployment on **Jetson Orin Nano**
  - Converted to ONNX for edge inference(have not included in the github repo)

---

## 🧠  Model Architecture

![Architecture](Picture1.png)

---

## 🔬 Model Comparisons

To evaluate the performance of different architectures, inference was also conducted using:

- **UForm Gen2 + Qwen 500M**
- **Gemini API (via cloud interface)**

Each model was tested on real-world VQA tasks from VizWiz for qualitative and latency-based comparison.

---

## 🔊 Voice I/O

- **Speech-to-Text**: [Whisper AI](https://github.com/openai/whisper)  (The provided code works on textual input from the user. The code can be modified to run using Whisper AI)
- **Text-to-Speech**: [gTTS (Google Text-to-Speech)](https://pypi.org/project/gTTS/)

---

## 🛠 Tech Stack

- `Python`
- `PyTorch`
- `Jetson JetPack SDK`
- `PaLI-Gemma 2`
- `UForm Gen2`, `Qwen 500M`
- `Gemini API`
- `Whisper AI`
- `gTTS`
- `ONNX`, `Quantization Aware Training`

---

### Requirements

- Jetson Orin Nano (JetPack 6+)
- Python 3.8+
- PyTorch with CUDA
- Dependencies: `transformers`, `onnxruntime`, `gtts`, `whisper`, etc.

---

You can refer to this PPT for more info [here](....)

## 📊 Results & Evaluation


| Model             | Accuracy (VizWiz) | Latency | Remarks                |
|------------------|------------------|---------|------------------------|
| PaLI-Gemma 2     | ⭐⭐⭐⭐☆             | Low     | Best edge inference    |
| UForm + Qwen 500M| ⭐⭐⭐☆☆             | Medium  | Generalized answers    |
| Gemini API       | ⭐⭐⭐⭐☆             | High    | Accurate but cloud-based |
