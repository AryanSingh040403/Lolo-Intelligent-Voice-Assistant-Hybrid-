# Stage 1: builder - Install dependencies and cache model weights
# Use a common, modern CUDA development image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip git curl libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# Install ML dependencies required for Qwen inference, RAG, and I/O caching
# Note: Specific versions are omitted for broader compatibility but should be pinned in setup_requirements.txt
RUN pip install torch transformers vllm peft bitsandbytes sentence-transformers
RUN pip install faster-whisper numpy coqui-tts

# Define cache locations for Hugging Face models
ENV HF_HOME=/app/cache
ENV VLLM_MODEL_REPO=Qwen/Qwen1.5-1.8B-Chat
ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV WHISPER_MODEL=small # Faster-whisper model
ENV XTTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# Create the cache directory
RUN mkdir -p $HF_HOME/models

# Script to pre-download the LLM, Embedding, and Whisper models during the build phase
WORKDIR /app
RUN echo "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    from sentence_transformers import SentenceTransformer; \
    import faster_whisper; \
    AutoModelForCausalLM.from_pretrained('$VLLM_MODEL_REPO'); \
    AutoTokenizer.from_pretrained('$VLLM_MODEL_REPO'); \
    SentenceTransformer('$EMBEDDING_MODEL'); \
    faster_whisper.WhisperModel('$WHISPER_MODEL', device='cpu', compute_type='int8');" > cache_model.py
RUN python3 cache_model.py

# Stage 2: runtime - Minimal image for serving application
# Use a minimal CUDA runtime image for deployment [3]
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

# Install necessary Python and audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy application files and requirements
WORKDIR /app
COPY scripts/. /app/scripts/
COPY deployment/vllm_start_server.sh /app/deployment/
COPY models/faiss_index_local /app/models/faiss_index_local/
COPY models/qwen_qlora_adapter /app/models/qwen_qlora_adapter/
COPY data/. /app/data/

# Reinstall necessary runtime Python packages (vLLM and others)
# Use requirements.txt or explicit installs for controlled environment
RUN pip install torch transformers vllm peft bitsandbytes sentence-transformers
RUN pip install faster-whisper pyaudio numpy coqui-tts sounddevice langchain-community langchain-openai

# Copy cached models from the builder stage [4, 5]
ENV HF_HOME=/app/cache
COPY --from=builder /app/cache $HF_HOME

# Set environment variables for vLLM
ENV CUDA_VISIBLE_DEVICES=0
ENV PATH="/usr/bin:$PATH"

# Expose port for the vLLM API server
EXPOSE 8000

# Set entrypoint to run the vLLM server
CMD ["/bin/bash", "/app/deployment/vllm_start_server.sh"]