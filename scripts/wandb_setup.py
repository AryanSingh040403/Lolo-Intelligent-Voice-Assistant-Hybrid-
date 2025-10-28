import wandb
import os
from langchain.callbacks.tracers import WandbTracer
from agent_core import QwenLLMAgent

# --- Configuration ---
WANDB_PROJECT_NAME = "Lolo_v2_Qwen_Agent_Tracing"

def setup_wandb_tracing() -> WandbTracer:
    """
    Initializes a Weights & Biases run and returns the WandbTracer instance 
    to log granular step-by-step traces of the LangChain Agent execution.[28]
    """
    if "WANDB_API_KEY" not in os.environ:
         print("Warning: WANDB_API_KEY environment variable not set. Tracing will run unlogged.")
    
    print("1. Initializing W&B Run and Tracer...")
    
    # Initialize the W&B run. 
    tracer = WandbTracer(
        run_args={
            "project": WANDB_PROJECT_NAME,
            "job_type": "LangChain_Agent_Run",
            "tags":,
        }
    )
    print(f"W&B Tracer initialized for project: {WANDB_PROJECT_NAME}")
    return tracer

def execute_agent_with_tracing(tracer: WandbTracer, agent: QwenLLMAgent):
    """
    Demonstrates agent execution across all three routing paths, logging the full trace.
    The tracer will automatically log:
    - LLM input/output (Thought process)
    - RAG retrieval events (on_retriever_end) including retrieved documents [29]
    - Tool execution events (on_tool_start/end) including inputs/outputs
    """
    
    config = {"callbacks": [tracer]} # Inject tracer via the config dictionary
    
    # --- Test 1: RAG Query (Triggers RAG Tool) ---
    rag_query = "Can you outline the quantization level and the context length used for the Qwen model?"
    print(f"\n--- Running Test 1: RAG Query ---")
    result_rag = agent.run(rag_query, config=config)
    print(f"Agent Response (RAG): {result_rag['output']}")

    # --- Test 2: Function Calling Query (Triggers Calculator Tool) ---
    tool_query = "If the memory usage is 2.9GB for 1.8B parameters, what is 888 multiplied by 3?"
    print(f"\n--- Running Test 2: Function Calling Query ---")
    result_tool = agent.run(tool_query, config=config)
    print(f"Agent Response (Tool): {result_tool['output']}")

    # --- Test 3: General Q&A Query (No Tool Use) ---
    general_query = "Why are LLMs called large language models?"
    print(f"\n--- Running Test 3: General Q&A Query ---")
    result_general = agent.run(general_query, config=config)
    print(f"Agent Response (General): {result_general['output']}")
    
    # Manually finish the W&B run
    if wandb.run is not None:
        wandb.run.finish()

# --- Main Execution Flow ---
if __name__ == "__main__":
    try:
        # Create a new agent instance
        qwen_agent = QwenLLMAgent()
        
        # Setup W&B Tracer
        tracer_instance = setup_wandb_tracing()
        
        # Execute tests
        execute_agent_with_tracing(tracer_instance, qwen_agent)
        print("\nMLOps tracing execution complete. Check W&B dashboard for traces.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you run `python rag_pipeline.py` first to create the FAISS index.")
    except Exception as e:
        print(f"Agent Execution Error: {e}")
        print("Ensure the vLLM server is running correctly on http://localhost:8000/v1.")