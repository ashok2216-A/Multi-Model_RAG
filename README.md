# File: README.md

# Document Processing and RAG Pipeline

This project extracts text and tables from various document types (`.pdf`, `.docx`, `.pptx`), processes the content, and ingests it into a Milvus vector database. It then uses a Retrieval-Augmented Generation (RAG) pipeline with Mistral AI to answer questions based on the document content.

## Project Structure

```
document-rag-pipeline/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
│
├── data/
│   └── mistral.pdf
│
├── output/
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── document_parser.py
    ├── processing_pipeline.py
    ├── text_utils.py
    └── vector_db.py
```

## Setup

### 1. Prerequisites
- Python 3.9+
- [Docker](https://www.docker.com/get-started) and Docker Compose (to run Milvus)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (ensure it's in your system's PATH or update the path in `src/config.py`)

### 2. Start Milvus
Run a Milvus instance using Docker Compose.
```bash
wget [https://github.com/milvus-io/milvus/releases/download/v2.4.4/milvus-standalone-docker-compose.yml](https://github.com/milvus-io/milvus/releases/download/v2.4.4/milvus-standalone-docker-compose.yml) -O docker-compose.yml
docker-compose up -d
```

### 3. Install Dependencies
Create a virtual environment and install the required packages.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root and add your Mistral API key:
```ini
MISTRAL_API_KEY="YOUR_MISTRAL_API_KEY_HERE"
```

## How to Run

### Step 1: Add Data
Place the documents you want to process into the `data/` directory.

### Step 2: Process and Ingest a Document
Run the main script with the `--process` flag, specifying the filename from the `data` directory.

```bash
python main.py --process "<input_file.pdf>"
```
This will:
1. Parse the document, extracting text and tables.
2. Save the structured output to the `output/` directory.
3. Connect to Milvus, embed the content, and insert it into the appropriate collections.

### Step 3: Query the Documents
Once the data is ingested, you can ask questions using the `--query` flag.

```bash
python main.py --query "What are the two main components of the Pixtral architecture?"
```

The script will retrieve relevant chunks from Milvus and generate an answer using Mistral AI.
