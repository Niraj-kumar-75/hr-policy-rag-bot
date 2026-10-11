import os
import shutil
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="HR Assistant RAG", page_icon="🏢")
st.title("🤖 Enterprise HR Policy Assistant")

PERSIST_DIR = "./chroma_db_v3"
if not os.path.exists("data"):
    os.makedirs("data")

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------------------------------------------
# 💡 FIX HERE: Safe DB Loader with Automatic Rebuild on KeyError
# -------------------------------------------------------------------
@st.cache_resource
def get_vector_db():
    embeddings = load_embedding_model()
    
    # 1. Safe Load from Disk
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        try:
            return Chroma(
                persist_directory=PERSIST_DIR, 
                embedding_function=embeddings,
                collection_name="hr_policies_v3"
            )
        except Exception:
            # Agar purana DB corrupt ho, toh folder ko safely clear kar denge
            st.warning("⚠️ Database cache corrupted. Rebuilding automatically...")
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    
    # 2. Fresh Build from PDFs
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    if not pdf_files:
        return None

    all_docs = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join("data", pdf))
            all_docs.extend(loader.load())
        except Exception:
            pass

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)

    if not chunks:
        return None

    return Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIR,
        collection_name="hr_policies_v3"
    )

# --- SIDEBAR UI ---
st.sidebar.header("📁 Document Management")
uploaded_file = st.sidebar.file_uploader("Upload HR Policy PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved: {uploaded_file.name}")

if st.sidebar.button("🔄 Re-Index / Clear Cache"):
    st.cache_resource.clear()
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    st.rerun()

vectorstore = get_vector_db()

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- SAFE LLM CALL ---
def ask_gemini_safely(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-3.8-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **Google Free API Rate Limit Reached (15 req/min).** Please wait 10 seconds and ask again."
        return f"❌ API Error: {e}"

# --- USER QUERY LOOP ---
if user_query := st.chat_input("Ask anything about company HR policies..."):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if vectorstore is None:
            st.warning("⚠️ No valid PDF found in 'data/' folder. Upload a PDF and click Re-Index.")
        else:
            with st.spinner("Searching policies..."):
                docs = vectorstore.similarity_search(user_query, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                
                prompt = f"""You are a helpful HR Assistant.
Answer the user's question accurately using ONLY the context provided below.
If the answer is not in the context, say "This policy is not mentioned in the provided document."

Context:
{context}

Question: {user_query}
Answer:"""

                answer_text = ask_gemini_safely(prompt)

                st.markdown(answer_text)

                with st.expander("📌 View Referenced Context"):
                    st.write(context)

                st.session_state.messages.append({"role": "assistant", "content": answer_text})