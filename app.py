"""
AHS-AI — RAG Chatbot (Improved with Fallbacks)
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

st.set_page_config(
    page_title="AHS-AI — RAG Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
footer, header { visibility: hidden; }
.stApp { background: linear-gradient(160deg, #0F0F1A 0%, #161630 40%, #1A1A2E 100%); }
section[data-testid="stSidebar"] {
    background: rgba(26, 26, 46, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(124, 58, 237, 0.15);
}
.stChatMessage {
    background: rgba(26, 26, 46, 0.6) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None


# ── Helper Functions ────────────────────────────────────────────────────────

def get_api_key():
    """Get API key."""
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY", "")


def test_embedding_models(api_key: str) -> str:
    """Test different embedding models and return the working one."""
    
    # Models to try in order
    models_to_try = [
        "models/embedding-001",
        "models/gemini-embedding-2",
        "text-embedding-004",
        "embedding-001",
    ]
    
    for model_name in models_to_try:
        try:
            # Try to create embeddings with this model
            test_embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key
            )
            # Test with a simple text
            test_embeddings.embed_query("test")
            st.session_state.embedding_model = model_name
            return model_name
        except Exception as e:
            continue
    
    return None


def load_and_process_pdf(pdf_path: str) -> list:
    """Load a PDF and split into chunks."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        return chunks
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return []


def create_vector_store(chunks: list, api_key: str, embedding_model: str):
    """Create vector store from document chunks."""
    if not chunks:
        return None
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key
        )
        
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            location=":memory:",
            collection_name="rag_docs",
        )
        return vector_store
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def get_answer(question: str, vector_store, api_key: str) -> tuple:
    """Get answer from RAG system."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        context_docs = retriever.invoke(question)
        
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            google_api_key=api_key
        )
        
        system_prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context. 
If the context doesn't contain the answer, say so honestly.

Context:
{context}"""
        
        messages = [
            ("system", system_prompt),
            ("human", question),
        ]
        
        response = llm.invoke(messages)
        return response.content, context_docs
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None, []


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 AHS-AI RAG Chatbot")
    st.divider()
    
    # API Key input
    api_key = get_api_key()
    
    if not api_key:
        st.warning("🔑 No API key found")
        api_key_input = st.text_input(
            "Enter your Google API Key",
            type="password",
            help="Get a free key at https://aistudio.google.com/app/apikey"
        )
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
            api_key = api_key_input
            st.rerun()
    else:
        st.success("✅ API key active")
        with st.expander("🔑 Change API Key"):
            new_api_key = st.text_input("New API Key", type="password")
            if new_api_key:
                os.environ["GOOGLE_API_KEY"] = new_api_key
                st.rerun()
    
    st.divider()
    
    # Test embedding models
    if api_key:
        if st.button("🧪 Test Embedding Models", help="Find which embedding models work with your API key"):
            with st.spinner("Testing embedding models..."):
                working_model = test_embedding_models(api_key)
                if working_model:
                    st.success(f"✅ Found working model: `{working_model}`")
                else:
                    st.error("""
                    ❌ No embedding models work with your API key.
                    
                    **Possible solutions:**
                    1. Enable billing in Google Cloud Console
                    2. Visit: https://console.cloud.google.com/billing
                    3. Enable API: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
                    4. Generate new key: https://aistudio.google.com/app/apikey
                    """)
    
    st.divider()
    
    # PDF upload
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files and api_key:
        if st.button("🚀 Process Documents", use_container_width=True):
            # Determine which embedding model to use
            if not st.session_state.embedding_model:
                with st.spinner("Finding working embedding model..."):
                    embedding_model = test_embedding_models(api_key)
                    if not embedding_model:
                        st.error("Could not find a working embedding model. Try the 'Test Embedding Models' button.")
                        st.stop()
            else:
                embedding_model = st.session_state.embedding_model
            
            with st.spinner("Processing PDFs..."):
                all_chunks = []
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_path = Path(temp_dir) / uploaded_file.name
                        temp_path.write_bytes(uploaded_file.getbuffer())
                        
                        chunks = load_and_process_pdf(str(temp_path))
                        all_chunks.extend(chunks)
                
                if all_chunks:
                    vector_store = create_vector_store(all_chunks, api_key, embedding_model)
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.docs_loaded = True
                        st.session_state.doc_count = len(all_chunks)
                        st.success(f"✅ Loaded {len(all_chunks)} chunks from {len(uploaded_files)} PDF(s)")
                        st.divider()
                        st.info(f"Using embedding model: `{embedding_model}`")
                        st.rerun()
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.warning("No valid PDFs to process")
    
    elif uploaded_files and not api_key:
        st.info("Enter API key above to process documents")
    
    st.divider()
    
    # Status
    if st.session_state.docs_loaded:
        st.markdown("### 📊 Status")
        st.metric("Chunks Loaded", st.session_state.doc_count)
        if st.session_state.embedding_model:
            st.caption(f"Model: {st.session_state.embedding_model}")
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.docs_loaded = False
            st.rerun()
    
    st.divider()
    st.caption("Powered by Gemini • LangChain • Qdrant")


# ── Main Content ────────────────────────────────────────────────────────────

st.markdown("# 📊 AHS-AI RAG Chatbot")
st.markdown("Ask questions about your uploaded PDF documents")
st.divider()

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** {doc.metadata.get('source', 'Unknown')}")
                    st.caption(doc.page_content[:200] + "...")

# Chat input
if st.session_state.docs_loaded:
    if prompt := st.chat_input("Ask a question about your documents..."):
        api_key = get_api_key()
        
        if not api_key:
            st.error("API key not set")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_answer(prompt, st.session_state.vector_store, api_key)
        
        if answer:
            st.markdown(answer)
            
            if sources:
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**[{i}]** {doc.metadata.get('source', 'Unknown')}")
                        st.caption(doc.page_content[:200] + "...")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

else:
    st.info("""
    👋 **Welcome to AHS-AI!**
    
    **Setup Steps:**
    1. Enter your Google API key (get one at https://aistudio.google.com/app/apikey)
    2. Click "🧪 Test Embedding Models" to verify your key works
    3. Upload PDF documents in the sidebar
    4. Click "🚀 Process Documents"
    5. Start asking questions!
    """)
