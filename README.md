# 🧠 Lolo v2.0 — Hybrid Low-Latency LLM Agent

A **production-grade hybrid LLM agent** designed for **real-time voice interaction**, combining **local cognition (RAG, tools)** with a **high‑performance LLM inference server**. The system is architected with **latency as a first‑class constraint**, achieving near‑human conversational fluency through concurrency, optimized model serving, and intelligent agent routing.

---

## 🚀 Project Overview

**Lolo v2.0** upgrades a traditional desktop voice assistant into a **cognitive hybrid AI agent** capable of:

* Real‑time speech understanding and response
* Intelligent decision‑making via agentic routing
* Grounded answers using local Retrieval‑Augmented Generation (RAG)
* Desktop and utility control through function calling

🔑 **Primary Objective:**

> **Low‑latency conversational fluency with Time‑to‑First‑Audio (TTFA) ≤ 500 ms**

This objective directly drives the system’s **concurrent, multi‑threaded pipeline** and **local-first architecture**.

---

## 🧠 Cognitive Architecture

The agent dynamically routes user intent through one of three execution paths:

1. **RAG Tooling** — For domain‑specific or document‑grounded queries
2. **Function Calling** — For deterministic desktop or utility actions
3. **LLM Reasoning** — For general conversational intelligence

This **agentic rerouting** ensures correctness, speed, and grounded responses while minimizing unnecessary LLM computation.

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

The system follows a **strict execution order** to ensure stability and performance:

1. **Environment Reset** — Clean virtual environment
2. **Dependency Installation** — Version‑pinned stable stack
3. **QLoRA Fine‑Tuning** — Function‑calling adapter training
4. **RAG Indexing** — Local FAISS index construction
5. **LLM Deployment** — vLLM GPU server via Docker
6. **Diagnostics** — Agent tracing & latency monitoring
7. **Live Execution** — Real‑time voice agent

The live agent uses a **producer–consumer concurrency model**, running **ASR, LLM inference, and TTS in parallel** to mask latency.

---

## 📊 Performance & Evaluation

Latency and correctness are treated as **core KPIs**:

| Metric                    | Dimension          | Target   |
| ------------------------- | ------------------ | -------- |
| **TTFA**                  | End‑to‑End Latency | ≤ 500 ms |
| **TTFT**                  | LLM Responsiveness | ≤ 350 ms |
| **Tool Call Accuracy**    | Cognitive Routing  | ≥ 95%    |
| **Response Groundedness** | RAG Quality        | ≥ 0.90   |

Latency tracing and agent decision paths are monitored via **Weights & Biases (W&B)**.

---

## 🧩 Data Preparation & RAG Design

* **Recursive, token‑aware chunking**
* Chunk size: **600 tokens**
* Overlap: **100 tokens**
* Optimized for **high recall** and **low retrieval latency**

FAISS indices are persisted locally to ensure **sub‑second similarity search** without network dependency.

---

## 🛠️ Prerequisites

* **Python:** ≤ 3.12 *(Python ≥ 3.13 is incompatible)*
* **Hardware:** NVIDIA GPU (required)
* **OS:** Linux / Windows (with manual audio driver setup)

> ⚠️ All dependencies **must** be installed using the pinned versions provided. Deviations may break compatibility.

---

## ⚠️ Limitations & Disclaimer

* **Dependency Fragility:** The stack relies on strict version pinning (bitsandbytes, vLLM, Coqui TTS)
* **Cold Start Cost:** Initial model downloads exceed 4GB; subsequent runs are cached
* **Audio I/O:** `pyaudio` and `sounddevice` may require manual system‑level configuration

---

## 📌 Why This Project Matters

This project demonstrates:

* Systems‑level thinking for **real‑time AI**
* Practical **LLM optimization and deployment**
* Agentic reasoning beyond simple prompt pipelines
* Production‑style monitoring and evaluation

It is designed as a **foundation for research, open‑source extension, and real‑world GenAI systems**.

---

## 🤝 Contributions

Contributions, discussions, and improvements are welcome. Feel free to open an issue or submit a pull request.
