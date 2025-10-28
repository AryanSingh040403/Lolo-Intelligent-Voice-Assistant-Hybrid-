#!/bin/bash
# Script to launch the vLLM OpenAI-compatible API server for the Qwen Agent

MODEL_NAME="Qwen/Qwen1.5-1.8B-Chat"
QUANTIZATION_TYPE="awq" # Assuming AWQ/GPTQ quantized version is used for production low-latency
DTYPE="bfloat16" # Use bfloat16 for computation if supported by GPU

echo "Starting vLLM server for ${MODEL_NAME} with quantization: ${QUANTIZATION_TYPE}"

# The vLLM server runs the LLM, listening for requests from agent_core.py
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --quantization ${QUANTIZATION_TYPE} \
    --dtype ${DTYPE} \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32000 \
    --enable-tools
    
echo "vLLM server stopped."