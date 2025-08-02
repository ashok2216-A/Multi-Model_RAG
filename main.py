import os
import argparse
from src.processing_pipeline import smart_file_processing
from src.vector_db import MilvusManager
from src.config import DATA_DIR


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run_processing(file_name):
    """Runs the document processing and ingestion pipeline."""
    input_file_path = os.path.join(DATA_DIR, file_name)
    
    # Step 1: Process the file and save JSON output
    smart_file_processing(input_file_path)
    
    # Step 2: Initialize Milvus and ingest the data
    milvus_manager = MilvusManager()
    milvus_manager.ingest_data()

def run_query(query):
    """Asks a question to the RAG system."""
    if not query:
        print("❌ Please provide a query with the --query flag.")
        return

    milvus_manager = MilvusManager()
    answer, chunks = milvus_manager.rag_answer(query)

    print("\\n" + "-"*50)
    print(f"❓ Query: {query}")
    print(f"💡 Answer: {answer}")
    print("\\n🔍 Retrieved Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\\nChunk {i}:")
        print(f"  Source: {chunk['source']} (Page {chunk['page_no']})")
        print(f"  Type: {chunk['type']}")
        print(f"  Similarity: {chunk['similarity_score']:.2f}")
        print(f"  Content: {chunk['content'][:200]}...")
    print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Processing and RAG Pipeline.")
    parser.add_argument("--process", type=str, help="Name of the file in the 'data' folder to process and ingest.")
    parser.add_argument("--query", type=str, help="A question to ask the RAG system.")

    args = parser.parse_args()

    if args.process:
        run_processing(args.process)
    elif args.query:
        run_query(args.query)
    else:
        print("Please specify an action: --process <filename> or --query \"<your question>\"")