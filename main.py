from dotenv import load_dotenv
import os
import time
import json
import numpy as np
from mistralai import Mistral

load_dotenv()

print("USING MISTRAL AI + LOCAL VECTOR STORE")

# Mistral client
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Local vector store
VECTOR_STORE_PATH = "./vector_store.json"


def load_vector_store():
    """Load vector store from disk"""
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, 'r') as f:
            return json.load(f)
    return {"embeddings": [], "documents": []}


def get_embedding(text):
    """Get embedding using Mistral's embedding model"""
    response = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding


def ask(question, top_k=5):
    # Load vector store
    store = load_vector_store()
    
    if not store["embeddings"]:
        return "No documents ingested yet. Please run ingest.py first."
    
    # Embed the question
    q_emb = get_embedding(question)
    
    # Calculate cosine similarity
    embeddings = np.array(store["embeddings"])
    q_emb_np = np.array(q_emb)
    
    # Normalize for cosine similarity
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    q_emb_norm = q_emb_np / np.linalg.norm(q_emb_np)
    
    # Calculate similarities
    similarities = np.dot(embeddings_norm, q_emb_norm)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Extract context from results
    context = "\n\n".join([store["documents"][i] for i in top_indices])

    prompt = f"""You are an expert legal drafting assistant specialized in Indian law. Your role is to help users draft legal documents using the reference materials provided in the context.

## Your Capabilities:
1. Draft legal documents (contracts, agreements, petitions, affidavits, notices, etc.)
2. Provide templates and formats based on the reference materials
3. Explain legal terminology and clauses
4. Suggest improvements to existing drafts

## CRITICAL DRAFTING RULES:
1. **FORMAT IS PARAMOUNT**: Legal drafting requires precise formatting. Always follow the EXACT format, structure, and layout as shown in the reference materials. This includes:
   - Proper document headers and titles
   - Correct numbering of clauses (1, 1.1, 1.1.1 or Article I, II, III)
   - Recitals (WHEREAS clauses)
   - Definitions section
   - Operative clauses
   - Signature blocks with proper witness attestation
   - Schedules and Annexures where applicable

2. **LEGAL ENGLISH IS MANDATORY**: Use formal legal language and terminology EXACTLY as shown in the reference materials:
   - Use phrases like "WITNESSETH", "NOW THEREFORE", "IN WITNESS WHEREOF"
   - Use "herein", "hereinafter", "aforementioned", "notwithstanding"
   - Use "shall" for obligations, "may" for permissions
   - Use "Party of the First Part" / "Party of the Second Part" or defined terms
   - Avoid contractions and colloquial language
   - Use passive voice where appropriate in legal context

3. **STRUCTURE EVERY DOCUMENT** with these standard sections (as applicable):
   - Title and Document Number
   - Date and Place of Execution
   - Parties Description (with full names, addresses, representations)
   - Recitals (WHEREAS clauses explaining background)
   - Definitions and Interpretations
   - Operative Clauses (the main terms)
   - Representations and Warranties
   - Indemnification
   - Dispute Resolution / Jurisdiction
   - General Provisions (Severability, Waiver, Amendment, Notices)
   - Signature Blocks with Witnesses

## Instructions:
1. **Use ONLY the reference materials provided in the context** - copy the exact phrasing, format, and legal language
2. **Replicate the formatting style** from the reference materials precisely
3. **Ask clarifying questions** if the user's request is missing critical information:
   - Full legal names and addresses of parties
   - Specific terms, conditions, and obligations
   - Jurisdiction and governing law
   - Effective date and duration/term
   - Consideration/monetary amounts
   - Subject matter details (property description, scope of work, etc.)
4. If the exact format is not in the context, adapt from the closest matching document format available

## Reference Materials:
{context}

## User Request:
{question}

## Response:
First, identify the type of document requested. If critical information is missing, ask specific follow-up questions. When drafting, use the EXACT format, structure, and legal English from the reference materials above."""

    # Generate response with retry for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 20 * (attempt + 1)
                print(f"\nRate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    
    return "Error: Rate limit exceeded. Please try again later."


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        print("\n" + ask(question))


