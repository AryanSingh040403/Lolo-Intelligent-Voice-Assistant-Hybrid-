# 🧠 Lolo v2.0 — Hybrid Low-Latency LLM Agent

A **production-grade hybrid LLM agent** engineered for **real-time voice interaction**, combining **local cognition (RAG + tools)** with a **high-performance GPU-backed LLM inference server**. The system is designed with **latency as a first-class constraint**, achieving near-human conversational fluency through concurrency, optimized model serving, and intelligent agent routing.

---

## 🚀 Project Overview

**Lolo v2.0** represents the evolution of a monolithic desktop voice assistant into a **modular, MLOps-ready cognitive AI system** capable of:

* Real-time speech-to-speech interaction
* Agentic decision-making with dynamic tool routing
* Grounded, hallucination-resistant answers using local RAG
* Deterministic desktop and utility control via function calling

🔑 **Primary Design Objective**

> **Low-latency conversational fluency with Time-to-First-Audio (TTFA) ≤ 500 ms**

This constraint directly shapes the system’s **concurrent, multi-threaded architecture** and **local-first execution strategy**.

---

## 🧠 Cognitive Architecture

At runtime, the agent performs **intent-aware routing**, dynamically selecting the optimal execution path:

1. **RAG Tooling** — For domain-specific, document-grounded queries
2. **Function Calling** — For deterministic system or utility actions
3. **LLM Reasoning** — For open-ended conversational intelligence

This **agentic rerouting mechanism** minimizes unnecessary LLM computation while maximizing **accuracy, speed, and response groundedness**.

---

## ⚙️ Technology Stack

### 🧠 Core AI Models

![Qwen](https://img.shields.io/badge/LLM-Qwen1.5--1.8B--Chat-4B0082?style=for-the-badge\&logo=openai\&logoColor=white)
![MiniLM](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-0A66C2?style=for-the-badge)
![Whisper](https://img.shields.io/badge/ASR-faster--whisper-FF6F00?style=for-the-badge)
![XTTS](https://img.shields.io/badge/TTS-Coqui--XTTS--v2.2-8A2BE2?style=for-the-badge)

### ⚙️ Frameworks & Systems

![vLLM](https://img.shields.io/badge/vLLM-High--Throughput--Serving-006400?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-Agentic--Routing-2F855A?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector--Search-0467DF?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-GPU--Deployment-2496ED?style=for-the-badge\&logo=docker\&logoColor=white)

### 🧪 MLOps & Deployment

![QLoRA](https://img.shields.io/badge/QLoRA-4--bit--Quantization-B83280?style=for-the-badge)
![PEFT](https://img.shields.io/badge/PEFT-Adapter--Training-6A5ACD?style=for-the-badge)
![W\&B](https://img.shields.io/badge/Weights%20%26%20Biases-Monitoring-FFBE00?style=for-the-badge)
![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU--Required-76B900?style=for-the-badge\&logo=nvidia\&logoColor=white)

---

## 🔄 Inference & Execution Pipeline

The system follows a **strict, deterministic execution order** to guarantee stability and reproducibility:

1. **Environment Reset** — Clean virtual environment
2. **Dependency Installation** — Version-pinned stable stack
3. **QLoRA Fine-Tuning** — Function-calling adapter training
4. **RAG Indexing** — Local FAISS index construction
5. **LLM Deployment** — vLLM GPU server via Docker
6. **Diagnostics** — Agent tracing & latency monitoring
7. **Live Execution** — Real-time voice agent

The live agent uses a **producer–consumer concurrency model**, executing **ASR, LLM inference, and TTS in parallel** to mask end-to-end latency.

---

## 📊 Performance & Evaluation

Latency and correctness are treated as **first-class KPIs**, validated via **Weights & Biases (W&B)** tracing:

| Metric                    | Dimension             | Target    |
| ------------------------- | --------------------- | --------- |
| **TTFA**                  | End-to-End Latency    | ≤ 500 ms  |
| **TTFT**                  | LLM Responsiveness    | ≤ 350 ms  |
| **Tokens Per Second**     | Generation Throughput | ≥ 100 TPS |
| **Tool Call Accuracy**    | Cognitive Routing     | ≥ 95%     |
| **Response Groundedness** | RAG Answer Quality    | ≥ 0.90    |

---

## 🧩 Data Preparation & RAG Design

* **Recursive, token-aware document chunking**
* Chunk size: **600 tokens**
* Overlap: **100 tokens**
* Optimized for **high recall** and **sub-second retrieval latency**

FAISS indices are persisted locally to ensure **offline-capable, low-latency vector search**.

---

## 🛠️ Prerequisites

* **Python:** ≤ 3.12 *(Python ≥ 3.13 is incompatible)*
* **Hardware:** NVIDIA GPU (required)
* **OS:** Linux / Windows (manual audio driver configuration may be required)

> ⚠️ All dependencies **must** be installed using the pinned versions provided. Deviations may break compatibility between `bitsandbytes`, `vLLM`, and `Coqui TTS`.

---

## ⚠️ Limitations & Disclaimer

* **Dependency Fragility:** The system relies on strict version pinning
* **Cold Start Cost:** Initial setup downloads exceed 4GB; subsequent runs are cached
* **Audio I/O:** `pyaudio` and `sounddevice` may require system-level configuration

---

## 📌 Why This Project Matters

This project demonstrates:

* Systems-level engineering for **real-time AI agents**
* Practical **LLM optimization, quantization, and GPU serving**
* Agentic reasoning beyond simple prompt pipelines
* Production-style monitoring and evaluation

It is designed as a **foundation for research, open-source extension, and real-world GenAI deployment**.

---

## 🤝 Contributions

Contributions, discussions, and improvements are welcome. Please open an issue or submit a pull request.
MODEL_NAME="Qwen/Qwen1.5-1.8B-Chat"
