import streamlit as st
from dotenv import load_dotenv
import os
import json
import numpy as np
from mistralai import Mistral

load_dotenv()

# Page config
st.set_page_config(
    page_title="Lawxygen - Legal Drafting Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Initialize Mistral client
@st.cache_resource
def get_mistral_client():
    return Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

mistral_client = get_mistral_client()

# Vector store path
VECTOR_STORE_PATH = "./vector_store.json"


@st.cache_data
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


def search_documents(question, top_k=5):
    """Search for relevant documents"""
    store = load_vector_store()
    
    if not store["embeddings"]:
        return []
    
    q_emb = get_embedding(question)
    
    embeddings = np.array(store["embeddings"])
    q_emb_np = np.array(q_emb)
    
    # Normalize for cosine similarity
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    q_emb_norm = q_emb_np / np.linalg.norm(q_emb_np)
    
    similarities = np.dot(embeddings_norm, q_emb_norm)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "text": store["documents"][i],
            "score": float(similarities[i])
        })
    return results


def generate_response(question, context):
    """Generate response using Mistral"""
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

    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #1E3A5F;
    }
    .score-badge {
        background-color: #1E3A5F;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">⚖️ Lawxygen</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Legal Document Drafting Assistant</p>', unsafe_allow_html=True)

# Instructions
with st.expander("📋 How to use this assistant"):
    st.markdown("""
    **This assistant helps you draft legal documents based on reference materials.**
    
    **You can ask for:**
    - Drafting contracts, agreements, NDAs, MOUs
    - Legal notices and petitions
    - Affidavits and declarations
    - Rent agreements, sale deeds
    - Power of Attorney documents
    - And more...
    
    **Tips for best results:**
    - Be specific about the type of document you need
    - Provide details like party names, dates, amounts when possible
    - The assistant will ask follow-up questions if more information is needed
    """)

# Check if vector store exists
store = load_vector_store()
if not store["embeddings"]:
    st.warning("⚠️ No documents ingested yet. Please run `python ingest.py` first to upload your legal documents.")
    st.stop()

st.success(f"✅ {len(store['documents'])} document chunks loaded")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
    show_sources = st.checkbox("Show source documents", value=True)
    
    st.divider()
    st.header("📊 Statistics")
    st.metric("Document Chunks", len(store['documents']))
    st.metric("Embedding Dimensions", len(store['embeddings'][0]) if store['embeddings'] else 0)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** (Relevance: {source['score']:.2%})")
                    st.markdown(f"```\n{source['text'][:500]}...\n```")

# Chat input
if prompt := st.chat_input("Ask me to draft a legal document (e.g., 'Draft an NDA between two companies')..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            # Search for relevant documents
            results = search_documents(prompt, top_k)
            context = "\n\n".join([r["text"] for r in results])
            
            # Generate response
            response = generate_response(prompt, context)
            st.markdown(response)
            
            # Show sources
            if show_sources and results:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(results):
                        st.markdown(f"**Source {i+1}** (Relevance: {source['score']:.2%})")
                        st.markdown(f"```\n{source['text'][:500]}...\n```")
    
    # Add assistant message to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": results
    })

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Built with Mistral AI & Streamlit</p>", unsafe_allow_html=True)
