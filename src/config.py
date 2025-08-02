import os

# Path Configuration
# This assumes Tesseract is installed within a .venv folder. Adjust if it's installed globally.
TESSERACT_PATH = os.path.join(os.getcwd(), ".venv", "Tesseract-OCR", "tesseract.exe")
os.environ["TESSERACT_PATH"] = TESSERACT_PATH

# Document Processing Configuration
DPI = 250  # Image resolution for OCR processing.
MIN_PAGE_CHARS = 10 # Minimum characters on a page to be considered searchable.

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Milvus Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
TEXT_COLLECTION_NAME = "textcollections"
TABLE_COLLECTION_NAME = "tablecollections"
EMBEDDING_DIM = 384 # Based on the 'all-MiniLM-L6-v2' model.

# File Paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
OCR_OUTPUT_PDF = os.path.join(OUTPUT_DIR, "ocr_output.pdf")
TEXT_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "text_output.json")
TABLES_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "tables_output.json")