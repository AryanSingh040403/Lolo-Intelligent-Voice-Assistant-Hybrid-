import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Configuration & Setup ---
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "../models/faiss_index_local"
DOCUMENTS_PATH = "../data/domain_docs" 

def setup_environment():
    """Ensures paths exist and creates a dummy document if none are present."""
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    if not os.path.exists(os.path.join(DOCUMENTS_PATH, "arch_notes.txt")):
        dummy_content = (
            "The core Lolo architecture transitioned from monolithic to an LLM Agent with MLOps. "
            "We chose the Qwen1.5-1.8B model due to its 32K context length and small 2.9GB memory footprint with Int4 quantization. "
            "The retrieval system uses FAISS and the all-MiniLM-L6-v2 model. "
            "Our chunking strategy is RecursiveCharacterTextSplitter at 600 tokens with 100 token overlap."
        )
        with open(os.path.join(DOCUMENTS_PATH, "arch_notes.txt"), "w", encoding="utf-8") as f:
            f.write(dummy_content)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    print("Environment setup complete. Created dummy document if needed.")

def load_and_split_documents() -> List:
    """Loads text data and splits it using token-aware Recursive Character Splitting."""
    # 1. Load data
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.txt", loader_cls=lambda path: open(path, encoding='utf-8').read())
    documents = loader.load()
    
    if not documents:
        print("No documents found to process.")
        return

    # 2. Initialize token-aware splitter [22, 21]
    # Using from_tiktoken_encoder ensures chunk size is measured in tokens.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", # Used for accurate token counting
        chunk_size=CHUNK_SIZE_TOKENS, 
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", " ", ""] # Recommended hierarchical separators [23]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")
    return chunks

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initializes and returns the Sentence-Transformer embedding model."""
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings

def create_and_save_vectorstore(chunks: List, embeddings: HuggingFaceEmbeddings):
    """Creates a new FAISS index and saves it to disk."""
    if not chunks:
        print("Cannot create vector store: No chunks available.")
        return

    print(f"Creating new FAISS index and generating {len(chunks)} embeddings...")
    # Create index from documents and embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index for persistence [8]
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved locally to {FAISS_INDEX_PATH}.")

# --- Pipeline Execution ---
if __name__ == "__main__":
    setup_environment()
    
    document_chunks = load_and_split_documents()
    embedding_model = get_embedding_model()
    
    create_and_save_vectorstore(document_chunks, embedding_model)
    print("\nFAISS Indexing pipeline complete. Ready for agent integration.")