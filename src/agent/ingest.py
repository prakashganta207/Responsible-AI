"""
Knowledge Base Ingestion Script
Primary Responsibility: Reads raw markdown, PDF, CSV, and Excel files, chunks them, generates semantic embeddings, and saves them to a FAISS index.
"""
import os
import re
import glob
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import uuid
import pandas as pd
from pypdf import PdfReader

# Add src to path to import telemetry
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.agentops.telemetry import tracker

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
VECTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vector_store")

def simple_chunker(text, max_words=100):
    """Simple word-based chunker."""
    words = str(text).split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def _is_number(v):
    try:
        float(str(v).replace(',', ''))
        return True
    except ValueError:
        return False

def table_sheet_to_sentences(df, report_name):
    """Turns a report-style sheet (title row, header row(s), data rows) into one sentence per data row,
    so each row carries its table title and column names and can be retrieved on its own."""
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    if len(df) < 3:
        return ""
    cells = df.astype(object).where(df.notna(), None).values.tolist()
    title = " ".join(dict.fromkeys(str(c).strip() for c in cells[0] if c is not None))  # merged cells repeat
    title = re.sub(r'\s+', ' ', title)

    # Header = first row after the title with at least 3 text cells; merge a second text header row if present
    h = next((i for i in range(1, min(6, len(cells)))
              if sum(1 for c in cells[i] if c is not None and not _is_number(c)) >= 3), None)
    if h is None:
        return ""
    top, headers = None, []
    for c in cells[h]:
        top = str(c).strip() if c is not None else top
        headers.append(top or "")
    start = h + 1
    if start < len(cells) and sum(1 for c in cells[start] if c is not None and not _is_number(c)) >= 2:
        headers = [f"{t} {s}".strip() if s is not None and not _is_number(s) else t
                   for t, s in zip(headers, cells[start])]
        start += 1
    headers = [re.sub(r'\s+', ' ', x) for x in headers]

    lines = []
    for row in cells[start:]:
        if all(c is None or _is_number(c) for c in row):
            continue  # column-numbering row (1, 2, 3, ...) or blank
        label = next(str(c).strip() for c in row if c is not None and not _is_number(c))
        values = [f"{hd}: {round(c, 3) if isinstance(c, float) else c}"
                  for hd, c in zip(headers, row)
                  if c is not None and _is_number(c) and hd and not re.match(r'(?i)s\.?\s*no|sl\.?\s*no', hd)]
        if values:
            lines.append(f"{report_name} - {title} - {label}: " + "; ".join(values) + ".")
    return "\n".join(lines)

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.md' or ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
        elif ext == '.xlsx':
            # Read every sheet (AISHE reports have 60+ tables); one sentence per data row
            report_name = os.path.splitext(os.path.basename(file_path))[0]
            sheets = pd.read_excel(file_path, sheet_name=None, header=None)
            parts = []
            for sheet_name, df in sheets.items():
                sentences = table_sheet_to_sentences(df, report_name)
                if not sentences:
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    if df.empty: continue
                    sentences = f"[Sheet: {sheet_name}]\n" + df.fillna('').to_string(index=False, header=False)
                parts.append(sentences)
            text = "\n\n".join(parts)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def ingest_documents():
    print("Starting document ingestion...")
    trace_id = str(uuid.uuid4())
    ingest_start = datetime.now()
    
    # 1. Load Model
    model_start = datetime.now()
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tracker.emit_span(
        trace_id=trace_id, span_type="load_embedding_model", 
        start_time=model_start, end_time=datetime.now(), metadata={"model": "all-MiniLM-L6-v2"}
    )
    
    # 2. Read and Chunk Documents
    chunk_start = datetime.now()
    all_chunks = []
    metadata = []
    
    all_files = glob.glob(os.path.join(DATA_DIR, "*.*"))
    for file_path in all_files:
        print(f"Processing: {os.path.basename(file_path)}")
        text = extract_text_from_file(file_path)
        if not text.strip(): continue
        
        filename = os.path.basename(file_path)
        chunks = simple_chunker(text, max_words=150)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({"source": filename, "chunk_index": i})
                
    tracker.emit_span(
        trace_id=trace_id, span_type="document_chunking", 
        start_time=chunk_start, end_time=datetime.now(),
        metadata={"total_files": len(all_files), "total_chunks": len(all_chunks)}
    )
    
    if not all_chunks:
        print("No documents found to ingest.")
        return

    # 3. Embed Chunks
    print(f"Embedding {len(all_chunks)} chunks (this may take a minute for large files)...")
    embed_start = datetime.now()
    embeddings = model.encode(all_chunks)
    tracker.emit_span(
        trace_id=trace_id, span_type="generate_embeddings", 
        start_time=embed_start, end_time=datetime.now(), metadata={"embedding_shape": str(embeddings.shape)}
    )
    
    # 4. Save to FAISS
    faiss_start = datetime.now()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTOR_DIR, "faiss.index"))
    
    # Save metadata
    with open(os.path.join(VECTOR_DIR, "metadata.json"), 'w') as f:
        json.dump({"chunks": all_chunks, "metadata": metadata}, f)
        
    tracker.emit_span(
        trace_id=trace_id, span_type="save_faiss_index", 
        start_time=faiss_start, end_time=datetime.now(), metadata={"index_path": "data/vector_store/faiss.index"}
    )

    print(f"Ingestion complete. Added {len(all_chunks)} chunks to FAISS index.")

if __name__ == "__main__":
    ingest_documents()
