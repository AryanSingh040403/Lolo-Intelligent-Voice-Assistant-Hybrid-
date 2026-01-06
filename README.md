
Lolo v2.0: Hybrid LLM Agent for Low-Latency Voice Interaction
This project documents the migration of a monolithic voice assistant (Lolo) to a modern, MLOps-ready architecture utilizing a local Qwen1.5-1.8B LLM, a private RAG pipeline, and a three-threaded concurrency model for real-time voice I/O.

1. Prerequisites and Setup
Environment: Ensure you have Python (3.10+) and a GPU with CUDA support for accelerated inference.

Virtual Environment: Create and activate a Python virtual environment (.venv).

Dependencies: Install libraries listed in scripts/setup_requirements.txt.

Weights & Biases (W&B): Set your API key environment variable for MLOps tracing.bash
export WANDB_API_KEY="your_api_key"

2. Execution Flow
Follow these steps to build the knowledge base and run the agent:

Fine-Tune Model (Optional but Recommended): Run the QLoRA script to specialize Qwen for function calling. The adapter weights are saved to models/qwen_qlora_adapter/.

Bash

python scripts/finetune_qwen_lora.py
Build RAG Index: Create the persistent FAISS index from documents in data/domain_docs/.

python scripts/rag_pipeline.py
Start LLM Server (Deployment): Launch the high-performance vLLM API endpoint, serving the quantized Qwen model.

bash deployment/vllm_start_server.sh
Run Real-Time Agent: Start the main application, which orchestrates STT, the LangChain Agent, and TTS streaming.

Bash

python scripts/realtime_agent.py
Validate MLOps Trace: Run the setup script to test the Agent's routing logic and log the trace to W&B Prompts.

Bash

python scripts/wandb_setup.py
3. MLOps Validation Metrics (W&B Dashboard)
To prove the success of this voice assistant upgrade, the AIML Engineer should monitor a blend of real-time latency and functional quality metrics.

Metric	Type	Target	Measurement & Significance
Time to First Audio (TTFA)	Latency (E2E)	≤500 ms	
Primary KPI for User Experience. Measures the total time from End-of-Speech (EOS) detection by faster-whisper until the first byte of synthesized audio is played by Coqui TTS.   

Time to First Token (TTFT)	Latency (LLM)	≤350 ms	
Time from vLLM request submission to the first token output. Measures the efficiency of the pre-fill and reasoning phase; monitored via WandbTracer on the LLM span.   

Tokens Per Second (TPS)	Latency (Throughput)	≥100 TPS	
The token generation rate during LLM streaming. High TPS ensures the model output consistently outpaces the TTS synthesis rate, preventing choppiness in audio output.   

Tool Call Accuracy	Functional	≥95%	
Measures the precision with which the fine-tuned Qwen model correctly identifies the required tool (RAG or calculator) and generates a valid, structured JSON argument payload. Monitored against ground truth data in W&B traces.   

Response Groundedness	Functional (RAG)	≥0.90	
Evaluates the factuality of the generated answer against the context retrieved by the RAG tool. A score of 1.0 means the answer is fully supported by the retrieved documents, validating the RAG component's output quality.   

=======
## Lolo_v2_Project
>>>>>>> 8103fa6efedf7dbd2ac075d65c05aea2a638e039
