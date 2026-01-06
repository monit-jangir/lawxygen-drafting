import streamlit as st
from dotenv import load_dotenv
import os
import json
import numpy as np
from mistralai import Mistral
import fitz  # PyMuPDF
import tempfile

load_dotenv()

# Page config
st.set_page_config(
    page_title="Lawxygen - Legal Drafting & Analysis",
    page_icon="",
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


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    os.unlink(tmp_path)
    return text


def summarize_document(text):
    """Generate a summary of the legal document"""
    prompt = f"""You are an expert legal document analyst. Analyze the following legal document and provide a comprehensive summary.

## Document:
{text[:15000]}  

## Provide the following analysis:

### 1. DOCUMENT TYPE & TITLE
Identify the type of legal document (e.g., Agreement, Contract, Deed, Notice, Petition, etc.)

### 2. EXECUTIVE SUMMARY
A brief 2-3 sentence overview of what this document is about.

### 3. PARTIES INVOLVED
List all parties mentioned in the document with their roles.

### 4. KEY TERMS & CONDITIONS
Summarize the main terms, obligations, and conditions in bullet points.

### 5. IMPORTANT DATES & DEADLINES
List any significant dates, durations, or deadlines mentioned.

### 6. FINANCIAL TERMS
Summarize any monetary amounts, payments, or financial obligations.

### 7. KEY CLAUSES
Highlight important clauses like:
- Termination
- Dispute Resolution
- Confidentiality
- Indemnification
- Jurisdiction

### 8. POTENTIAL CONCERNS / RED FLAGS
Note any unusual clauses or potential issues that should be reviewed carefully.

Provide the analysis in a clear, structured format."""

    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def answer_document_question(question, document_text, chat_history=""):
    """Answer questions about the uploaded document"""
    prompt = f"""You are an expert legal document analyst. Answer the user's question based ONLY on the document provided.

## Document Content:
{document_text[:15000]}

## Previous Conversation:
{chat_history}

## User Question:
{question}

## Instructions:
1. Answer based ONLY on the information in the document
2. If the information is not in the document, say "This information is not found in the document."
3. Quote specific sections when relevant
4. Be precise and cite clause numbers or sections when applicable
5. If the question is about legal interpretation, provide the factual content and suggest consulting a legal professional for interpretation

## Answer:"""

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
st.markdown('<p class="main-header">Lawxygen</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Legal Document Drafting & Analysis Assistant</p>', unsafe_allow_html=True)

# Create tabs for different features
tab1, tab2 = st.tabs(["Draft Documents", "Analyze & Summarize"])

# ==================== TAB 1: DRAFTING ====================
with tab1:
    # Check if vector store exists
    store = load_vector_store()
    if not store["embeddings"]:
        st.warning("No documents ingested yet. Please run `python ingest.py` first to upload your legal documents.")
    else:
        st.success(f"{len(store['documents'])} document chunks loaded")

    # Sidebar for drafting settings
    with st.sidebar:
        st.header("Drafting Settings")
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
        show_sources = st.checkbox("Show source documents", value=True)
        
        st.divider()
        st.header("Statistics")
        st.metric("Document Chunks", len(store['documents']) if store["embeddings"] else 0)
        st.metric("Embedding Dimensions", len(store['embeddings'][0]) if store['embeddings'] else 0)

    # Instructions
    with st.expander("How to use the Drafting Assistant"):
        st.markdown("""
        **This assistant helps you draft legal documents based on reference materials.**
        
        **Steps to draft a document:**
        1. Select the document type from the dropdown
        2. Fill in the party details, dates, and other required information
        3. Click "Generate Draft" to create your document
        4. Edit, download, or modify the generated draft
        5. Use the chat below for any follow-up questions or modifications
        """)
    
    # Quick Draft Prompt
    st.markdown("### Quick Draft")
    quick_prompt = st.text_area(
        "Describe what you want to draft:",
        placeholder="e.g., Draft an NDA between ABC Company and XYZ Corp for a software development project. The agreement should be for 2 years with a confidentiality period of 5 years...",
        height=100,
        key="quick_prompt"
    )
    
    if st.button("Generate from Prompt", type="primary"):
        if quick_prompt:
            with st.spinner("Generating your legal document..."):
                # Search for relevant templates
                results = search_documents(quick_prompt, top_k=5)
                context = "\n\n".join([r["text"] for r in results])
                
                # Generate the document
                draft = generate_response(quick_prompt, context)
            
            # Store the generated draft
            st.session_state.generated_draft = draft
            st.session_state.form_doc_type = "Custom Draft"
            st.rerun()
        else:
            st.error("Please enter a description of what you want to draft.")
    
    st.divider()
    st.markdown("### Or Fill in the Form Below")
    
    # Document type selection
    doc_type = st.selectbox(
        "Select Document Type",
        [
            "Non-Disclosure Agreement (NDA)",
            "Service Agreement",
            "Employment Contract",
            "Rent/Lease Agreement",
            "Memorandum of Understanding (MOU)",
            "Power of Attorney",
            "Partnership Deed",
            "Sale Deed",
            "Loan Agreement",
            "Consultancy Agreement",
            "Other (Specify in chat)"
        ]
    )
    
    st.divider()
    
    # Create form columns
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        st.markdown("### Agreement Details")
        date_of_agreement = st.text_input("Date of Agreement", placeholder="e.g., 5")
        month_of_agreement = st.selectbox("Month of Agreement", 
            ["January", "February", "March", "April", "May", "June", 
             "July", "August", "September", "October", "November", "December"])
        year_of_agreement = st.text_input("Year of Agreement", placeholder="e.g., 2026")
        place_of_agreement = st.text_input("Place of Agreement", placeholder="e.g., New Delhi")
        jurisdiction = st.text_input("Jurisdiction / Governing Law", placeholder="e.g., Courts of New Delhi")
        
        st.markdown("### Duration")
        effective_date = st.text_input("Effective Date", placeholder="e.g., 5th January 2026")
        term_duration = st.text_input("Term/Duration", placeholder="e.g., 2 years")
        
    with form_col2:
        st.markdown("### First Party Details")
        party1_name = st.text_input("First Party Name", placeholder="e.g., Romi Kumar")
        party1_type = st.selectbox("First Party Type", ["Individual", "Company", "Partnership Firm", "LLP", "Trust", "Other"])
        party1_father_name = st.text_input("Father's Name (if individual)", placeholder="e.g., Shri Ram Kumar")
        party1_address = st.text_area("First Party Address", placeholder="e.g., 123, ABC Street, New Delhi - 110001", height=80)
        party1_pan = st.text_input("First Party PAN Number", placeholder="e.g., ABCDE1234F")
        party1_email = st.text_input("First Party Email", placeholder="e.g., romi@example.com")
    
    st.divider()
    
    form_col3, form_col4 = st.columns(2)
    
    with form_col3:
        st.markdown("### Second Party Details")
        party2_name = st.text_input("Second Party Name", placeholder="e.g., Rabbit AI Pvt. Ltd.")
        party2_type = st.selectbox("Second Party Type", ["Company", "Individual", "Partnership Firm", "LLP", "Trust", "Other"])
        party2_incorporation_year = st.text_input("Year of Incorporation (if company)", placeholder="e.g., 2020")
        party2_address = st.text_area("Second Party Address", placeholder="e.g., 456, XYZ Tower, Mumbai - 400001", height=80)
        party2_pan = st.text_input("Second Party PAN/CIN Number", placeholder="e.g., U72200MH2020PTC123456")
        party2_authorized_rep = st.text_input("Authorized Representative Name", placeholder="e.g., Mr. John Doe")
        party2_designation = st.text_input("Representative Designation", placeholder="e.g., Director")
        
    with form_col4:
        st.markdown("### Financial Terms")
        consideration_amount = st.text_input("Consideration/Amount", placeholder="e.g., Rs. 5,00,000/-")
        consideration_words = st.text_input("Amount in Words", placeholder="e.g., Rupees Five Lakhs Only")
        payment_terms = st.text_area("Payment Terms", placeholder="e.g., 50% advance, 50% on completion", height=80)
        
        st.markdown("### Additional Terms")
        confidentiality_period = st.text_input("Confidentiality Period", placeholder="e.g., 3 years")
        notice_period = st.text_input("Notice Period", placeholder="e.g., 30 days")
    
    st.divider()
    
    # Additional clauses
    st.markdown("### Subject Matter / Scope")
    subject_matter = st.text_area(
        "Describe the subject matter or scope of work",
        placeholder="e.g., Development of mobile application with features including...",
        height=100
    )
    
    special_conditions = st.text_area(
        "Any Special Conditions or Additional Clauses",
        placeholder="Enter any special terms, conditions, or clauses you want to include...",
        height=100
    )
    
    st.divider()
    
    # Generate button
    if st.button("Generate Draft", type="primary", use_container_width=True):
        # Validate required fields
        if not party1_name or not party2_name:
            st.error("Please fill in at least the party names to generate a draft.")
        else:
            # Build the form data context
            form_data = f"""
DOCUMENT TYPE: {doc_type}

AGREEMENT DETAILS:
- Date: {date_of_agreement} {month_of_agreement} {year_of_agreement}
- Place of Execution: {place_of_agreement}
- Jurisdiction: {jurisdiction}
- Effective Date: {effective_date}
- Term/Duration: {term_duration}

FIRST PARTY DETAILS:
- Name: {party1_name}
- Type: {party1_type}
- Father's Name: {party1_father_name}
- Address: {party1_address}
- PAN: {party1_pan}
- Email: {party1_email}

SECOND PARTY DETAILS:
- Name: {party2_name}
- Type: {party2_type}
- Year of Incorporation: {party2_incorporation_year}
- Address: {party2_address}
- PAN/CIN: {party2_pan}
- Authorized Representative: {party2_authorized_rep}
- Designation: {party2_designation}

FINANCIAL TERMS:
- Consideration Amount: {consideration_amount}
- Amount in Words: {consideration_words}
- Payment Terms: {payment_terms}

ADDITIONAL TERMS:
- Confidentiality Period: {confidentiality_period}
- Notice Period: {notice_period}

SUBJECT MATTER / SCOPE:
{subject_matter}

SPECIAL CONDITIONS:
{special_conditions}
"""
            
            with st.spinner("Generating your legal document..."):
                # Search for relevant templates
                results = search_documents(doc_type, top_k=5)
                context = "\n\n".join([r["text"] for r in results])
                
                # Generate the document
                generation_prompt = f"""Generate a complete {doc_type} using the form data provided and following the exact format from the reference materials.

{form_data}

Use the EXACT legal format, structure, and language from the reference materials. Fill in all placeholders with the provided information. Include all standard clauses for this type of document."""
                
                draft = generate_response(generation_prompt, context)
            
            # Store the generated draft
            st.session_state.generated_draft = draft
            st.session_state.form_doc_type = doc_type
    
    # Display and edit the generated draft
    if "generated_draft" in st.session_state and st.session_state.generated_draft:
        st.divider()
        st.markdown("### Generated Draft")
        
        # Formatting toolbar info
        st.info("Use the text editor below to modify your draft. You can copy, edit, and format as needed.")
        
        # Editable text area with the draft
        edited_draft = st.text_area(
            "Edit your draft:",
            value=st.session_state.generated_draft,
            height=500,
            key="draft_editor"
        )
        
        # Action buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            if st.button("Copy to Clipboard"):
                st.code(edited_draft)
                st.success("Draft displayed above - use Ctrl+C to copy!")
        
        with col_btn2:
            # Download as text file
            st.download_button(
                label="Download as TXT",
                data=edited_draft,
                file_name=f"{st.session_state.form_doc_type.replace(' ', '_').replace('/', '_')}.txt",
                mime="text/plain"
            )
        
        with col_btn3:
            if st.button("Modify Draft"):
                modification = st.session_state.get("modification_request", "")
                if modification:
                    with st.spinner("Modifying draft..."):
                        mod_prompt = f"""Modify the following legal document based on this request: {modification}

Current Document:
{edited_draft}

Provide the complete modified document."""
                        results = search_documents(st.session_state.form_doc_type, top_k=3)
                        context = "\n\n".join([r["text"] for r in results])
                        st.session_state.generated_draft = generate_response(mod_prompt, context)
                        st.rerun()
        
        with col_btn4:
            if st.button("Clear Draft"):
                st.session_state.generated_draft = None
                st.rerun()
        
        # Modification request input
        st.text_input(
            "Request modifications to the draft:",
            placeholder="e.g., Add a non-compete clause for 1 year",
            key="modification_request"
        )
    
    # Chat section for follow-up questions
    st.divider()
    st.markdown("### Chat with Assistant")
    st.caption("Ask follow-up questions, request modifications, or get help with your draft")
    
    # Initialize chat history for drafting
    if "draft_messages" not in st.session_state:
        st.session_state.draft_messages = []

    # Display chat history
    for message in st.session_state.draft_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and show_sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (Relevance: {source['score']:.2%})")
                        st.markdown(f"```\n{source['text'][:500]}...\n```")

    # Chat input for drafting
    if prompt := st.chat_input("Ask a question or request modifications..."):
        if not store["embeddings"]:
            st.error("Please ingest documents first using `python ingest.py`")
        else:
            # Add user message to chat history
            st.session_state.draft_messages.append({"role": "user", "content": prompt})
            
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
                        with st.expander("View Sources"):
                            for i, source in enumerate(results):
                                st.markdown(f"**Source {i+1}** (Relevance: {source['score']:.2%})")
                                st.markdown(f"```\n{source['text'][:500]}...\n```")
            
            # Add assistant message to chat history
            st.session_state.draft_messages.append({
                "role": "assistant",
                "content": response,
                "sources": results
            })

# ==================== TAB 2: DOCUMENT ANALYSIS ====================
with tab2:
    st.subheader("Upload & Analyze Legal Documents")
    
    with st.expander("How to use Document Analysis"):
        st.markdown("""
        **Upload any legal document to:**
        - Get a comprehensive summary
        - Ask questions about the document
        - Understand key terms and clauses
        - Identify important dates and obligations
        
        **Supported formats:** PDF
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Store the uploaded file in session state
        if "uploaded_doc_name" not in st.session_state or st.session_state.uploaded_doc_name != uploaded_file.name:
            st.session_state.uploaded_doc_name = uploaded_file.name
            st.session_state.uploaded_doc_text = None
            st.session_state.doc_summary = None
            st.session_state.doc_messages = []
        
        # Extract text from PDF
        if st.session_state.uploaded_doc_text is None:
            with st.spinner("Extracting text from document..."):
                st.session_state.uploaded_doc_text = extract_text_from_pdf(uploaded_file)
        
        st.success(f"Document loaded: **{uploaded_file.name}** ({len(st.session_state.uploaded_doc_text):,} characters)")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Document Summary")
            
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Analyzing document and generating summary..."):
                    st.session_state.doc_summary = summarize_document(st.session_state.uploaded_doc_text)
            
            if st.session_state.doc_summary:
                st.markdown(st.session_state.doc_summary)
        
        with col2:
            st.subheader("Ask Questions")
            
            # Initialize document chat history
            if "doc_messages" not in st.session_state:
                st.session_state.doc_messages = []
            
            # Display document chat history
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.doc_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Question input
            question = st.text_input("Ask a question about this document:", key="doc_question")
            
            if st.button("Ask", type="secondary") and question:
                # Add user message
                st.session_state.doc_messages.append({"role": "user", "content": question})
                
                # Build chat history for context
                chat_history = "\n".join([
                    f"{m['role'].upper()}: {m['content']}" 
                    for m in st.session_state.doc_messages[-6:]  # Last 6 messages for context
                ])
                
                with st.spinner("Analyzing..."):
                    answer = answer_document_question(
                        question, 
                        st.session_state.uploaded_doc_text,
                        chat_history
                    )
                
                # Add assistant message
                st.session_state.doc_messages.append({"role": "assistant", "content": answer})
                
                # Rerun to show updated chat
                st.rerun()
            
            # Clear chat button
            if st.session_state.doc_messages:
                if st.button("Clear Chat"):
                    st.session_state.doc_messages = []
                    st.rerun()
    else:
        st.info("Upload a PDF document to get started with analysis")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("<p style='text-align: center; color: #666;'>Built with Mistral AI & Streamlit</p>", unsafe_allow_html=True)
