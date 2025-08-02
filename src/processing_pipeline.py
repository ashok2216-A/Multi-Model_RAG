import os
import json
import fitz
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, OCR_OUTPUT_PDF, DPI,
    TEXT_OUTPUT_JSON, TABLES_OUTPUT_JSON
)
from src.document_parser import (
    is_page_searchable, process_ocr, extract_tables_from_fitz_doc,
    extract_text_and_tables_from_docx, extract_text_and_tables_from_pptx
)
from src.text_utils import chunking_workflow, uniquify_columns

def create_elements_with_metadata(chunks, tables, input_file, text_positions=None, table_positions=None):
    """Combines text chunks and tables into a single list of elements with metadata."""
    elements = []
    base_name = os.path.basename(input_file)
    # Add text chunks with position if available
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        metadata.update({
            "source_document": base_name, "chunk_id": i, 
            "position": text_positions[i-1] if text_positions and i-1 < len(text_positions) else None})
        elements.append({"type": "text","content": chunk['text'],"metadata": metadata,"token_count": chunk['token_count']})

    # Add tables with position if available
    for j, table in enumerate(tables, start=i+1):
        if isinstance(table, dict) and "metadata" in table:
            # PDF table with metadata
            meta = table["metadata"].copy()
            meta.update({"source_document": base_name,"chunk_id": j})
            content = table.get("content", table)
        else:
            meta = {
                "source_document": base_name,"chunk_id": j, 
                "position": table_positions[j - (i+1)] if table_positions and (j - (i+1)) < len(table_positions) else j}
            content = table
        elements.append({"type": "table","content": content,"metadata": meta})

    return elements

def process_pdf_pages(input_pdf):
    """Processes each page of a PDF, performs OCR if needed, and extracts text and tables."""
    src = fitz.open(input_pdf)
    ocr_doc = fitz.open()
    tables, text_blocks = [], []
    text_positions, table_positions = [], []
    filename = os.path.basename(input_pdf)

    for i, page in enumerate(src):
        print(f"   📄 Processing Page {i + 1}/{len(src)}")
        searchable = is_page_searchable(page)
        temp_doc = None  # To manage OCR temp doc lifetime
        if not searchable:
            print(f"      🔎 Searchable: ❌ No, requires OCR")
            temp_doc = process_ocr(page, dpi=DPI)
            blocks_page = temp_doc[0]
            page_tables = extract_tables_from_fitz_doc(temp_doc, 0)
            ocr_doc.insert_pdf(temp_doc)
        else:
            print(f"      🔎 Searchable: ✅ Yes")
            blocks_page = page
            page_tables = extract_tables_from_fitz_doc(src, i)

        for idx, (bbox, df) in enumerate(page_tables, 1):
            df.columns = uniquify_columns(df.columns.astype(str))
            tables.append({
                "type": "table",
                "content": df.to_dict(orient="records"),
                "metadata": {"page_number": i + 1,"columns": df.columns.tolist(),
                    "table_index_on_page": idx,"position": bbox[1], "source_document": filename}})
            table_positions.append(bbox[1])  # Save y-position

        table_boxes = [box for box, _ in page_tables]
        for block in blocks_page.get_text("blocks"):
            if len(block) >= 5:
                x0, y0, x1, y1, text = block[:5]
                rect = fitz.Rect(x0, y0, x1, y1)
                if not any(fitz.Rect(*box).intersects(rect) for box in table_boxes) and text.strip():
                    text_blocks.append((i, y0, text.strip()))
                    text_positions.append(y0)  # Save y-position

    text_blocks.sort(key=lambda x: (x[0], x[1]))
    full_text = "\n".join(text for _, _, text in text_blocks)
    chunks = chunking_workflow(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_positions = text_positions[:len(chunks)]
    elements = create_elements_with_metadata(
        chunks, [t for t in tables], input_pdf, text_positions=chunk_positions, table_positions=table_positions
    )
    return src, ocr_doc, elements


def save_processed_output(elements, ocr_doc):
    """Saves extracted text chunks and tables to JSON files."""
    texts = [e for e in elements if e['type'] == 'text']
    tables = [e for e in elements if e['type'] == 'table']

    os.makedirs(os.path.dirname(TEXT_OUTPUT_JSON), exist_ok=True)

    if texts:
        with open(TEXT_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"text_chunks": texts}, f, indent=2, ensure_ascii=False)
    if tables:
        with open(TABLES_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"tables": tables}, f, indent=2, ensure_ascii=False)
            
    if ocr_doc and len(ocr_doc) > 0:
        ocr_doc.save(OCR_OUTPUT_PDF)

def smart_file_processing(input_file, ocr_output_pdf="ocr_output.pdf"):
    """Main function to process a file based on its extension."""
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    ext = os.path.splitext(input_file)[1].lower()
    print(f"📁 Processing file: {input_file}")
    
    src, ocr_doc, elements = None, None, []
    
    if ext == ".pdf":
        src, ocr_doc, elements = process_pdf_pages(input_file)
    elif ext == ".docx":
        texts, tables = extract_text_and_tables_from_docx(input_file)
        full_text = "\\n".join(texts)
        chunks = chunking_workflow(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        elements = create_elements_with_metadata(chunks, tables, input_file)
    elif ext == ".pptx":
        texts, tables = extract_text_and_tables_from_pptx(input_file)
        full_text = "\\n".join(texts)
        chunks = chunking_workflow(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        elements = create_elements_with_metadata(chunks, tables, input_file)
    else:
        print(f"❌ Unsupported file type: {ext}")
        return

    save_processed_output(elements, ocr_doc)
    
    if src: src.close()
    if ocr_doc: ocr_doc.close()

    print(f"\n✅ Processing done! Output saved to '{os.path.dirname(TEXT_OUTPUT_JSON)}/' directory.")