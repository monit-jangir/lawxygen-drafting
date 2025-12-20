from dotenv import load_dotenv
import os
import json
import time
from mistralai import Mistral
import fitz  # PyMuPDF

load_dotenv()

# Mistral client
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Local vector store path
VECTOR_STORE_PATH = "./vector_store.json"


def get_embedding(text, max_retries=3):
    """Get embedding using Mistral's embedding model with retry"""
    for attempt in range(max_retries):
        try:
            response = mistral_client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 30 * (attempt + 1)
                print(f"\n  Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest_pdf(pdf_path):
    """Extract, chunk, embed, and store PDF in local vector store"""
    print(f"Reading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Check for existing progress
    embeddings = []
    start_idx = 0
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, 'r') as f:
            existing = json.load(f)
            if existing.get("embeddings"):
                embeddings = existing["embeddings"]
                start_idx = len(embeddings)
                print(f"Resuming from chunk {start_idx + 1}...")
    
    print("Embedding and storing chunks...")
    for i in range(start_idx, len(chunks)):
        embedding = get_embedding(chunks[i])
        embeddings.append(embedding)
        print(f"  Embedded chunk {i+1}/{len(chunks)}")
        
        # Save progress every 10 chunks
        if (i + 1) % 10 == 0:
            store = {"embeddings": embeddings, "documents": chunks[:len(embeddings)]}
            with open(VECTOR_STORE_PATH, 'w') as f:
                json.dump(store, f)
            print(f"  Progress saved...")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    # Final save
    store = {
        "embeddings": embeddings,
        "documents": chunks
    }
    with open(VECTOR_STORE_PATH, 'w') as f:
        json.dump(store, f)
    
    print(f"\nDone! Stored {len(chunks)} chunks in {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    pdf_path = "Drafting case material -2025.pdf"
    if os.path.exists(pdf_path):
        ingest_pdf(pdf_path)
    else:
        print(f"PDF not found: {pdf_path}")
        print("Please place your PDF file in the project directory.")
