import os
import streamlit as st
from src.processing_pipeline import smart_file_processing
from src.vector_db import MilvusManager
from src.config import DATA_DIR

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 Document RAG Assistant")
st.markdown("Upload a document (`.pdf`, `.docx`, `.pptx`), and I'll answer your questions based on its content.")

# --- Initialize Milvus Manager ---
# Use a singleton pattern to avoid re-initializing on every interaction.
@st.cache_resource
def get_milvus_manager():
    """Initializes and returns the MilvusManager instance."""
    try:
        return MilvusManager()
    except Exception as e:
        st.error(f"Failed to connect to Milvus. Please ensure Milvus is running. Error: {e}")
        return None

milvus_manager = get_milvus_manager()

# --- Sidebar for File Upload and Processing ---
with st.sidebar:
    st.header("1. Process a Document")
    
    uploaded_file = st.file_uploader(
        "Upload a .pdf, .docx, or .pptx file",
        type=["pdf", "docx", "pptx"]
    )

    if uploaded_file is not None:
        # Save the uploaded file to the data directory
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

        if st.button("Process Document"):
            if not milvus_manager:
                st.stop()
            
            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                try:
                    # Step 1: Process the file and save JSON output
                    smart_file_processing(file_path)
                    st.success("✅ Document parsing and chunking complete.")
                    
                    # Step 2: Ingest the data into Milvus
                    milvus_manager.ingest_data()
                    st.success("✅ Data ingested into Milvus successfully.")
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

# --- Main Area for Q&A ---
st.header("2. Ask a Question")

if not milvus_manager:
    st.warning("Cannot proceed with Q&A. Milvus connection failed.")
else:
    query = st.text_input(
        "Enter your question about the document:",
        placeholder="e.g., What are the key findings?"
    )

    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching for answers..."):
                try:
                    answer, chunks = milvus_manager.rag_answer(query)
                    
                    st.subheader("💡 Answer")
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Retrieved Chunks"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"---")
                            st.markdown(f"**Chunk {i}:**")
                            st.markdown(f"**Source:** {chunk.get('source', 'N/A')} (Page {chunk.get('page_no', 'N/A')})")
                            st.markdown(f"**Type:** {chunk.get('type', 'N/A')}")
                            st.markdown(f"**Similarity Score:** {chunk.get('similarity_score', 0):.4f}")
                            st.text_area(
                                label=f"Content of Chunk {i}",
                                value=chunk.get('content', 'No content available.'),
                                height=200,
                                key=f"chunk_{i}"
                            )
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")