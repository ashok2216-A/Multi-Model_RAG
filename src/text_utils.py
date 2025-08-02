import ftfy
import tiktoken
import collections

def process_text_utf8(text):
    """Cleans and standardizes text to UTF-8."""
    if not isinstance(text, str):
        text = str(text)
    text = ftfy.fix_text(text)
    text = text.replace("\x00", "").replace("\ufeff", "")
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = text.replace('\n', ' ').replace('\"', '')
    return text.strip()

def count_tokens(text):
    """Counts the number of tokens in a text string."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def custom_chunking(text, chunk_size, overlap):
    """Splits text into chunks based on token count."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_token_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_token_ids))
        start += chunk_size - overlap
    return chunks

def chunking_workflow(text, max_tokens, overlap):
    """Processes, cleans, and chunks text, returning structured data."""
    clean_text = process_text_utf8(text)
    if not clean_text:
        return []
    
    token_count = count_tokens(clean_text)
    if token_count <= max_tokens:
        return [{"text": clean_text, "metadata": {}, "token_count": token_count}]
    
    chunks = custom_chunking(clean_text, chunk_size=max_tokens, overlap=overlap)
    return [
        {"text": chunk, "metadata": {}, "token_count": count_tokens(chunk)}
        for chunk in chunks
    ]

def uniquify_columns(cols):
    """Makes column names in a DataFrame unique by appending numbers."""
    counts = collections.Counter()
    new_cols = []
    for col in cols:
        counts[col] += 1
        new_cols.append(col if counts[col] == 1 else f"{col}_{counts[col] - 1}")
    return new_cols