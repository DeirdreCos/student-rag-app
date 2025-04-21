import os
import streamlit as st
from langchain.document_loaders.pypdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit.components.v1 as components
import re

st.set_page_config(page_title="📚 Student RAG MVP", layout="wide")
st.title("📖 Research Assistant with Inline PDF Viewer")

# — 1. API Key —
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# — 2. PDF Uploader & Save to Disk —
uploaded_files = st.file_uploader(
    "Upload PDF(s) from your reading list", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for pdf in uploaded_files:
        # Save to disk
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        # Load & chunk
        loader = PyPDFLoader(pdf.name)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        for c in chunks:
            c.metadata["source"] = pdf.name
            c.metadata["page_number"] = c.metadata.get("page", c.metadata.get("page_number"))
            docs.append(c)

    st.success(f"Embedded {len(docs)} chunks from {len(uploaded_files)} files")

    # — 3. Build Vector Store —
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=None)

    # — 4. Search UI —
    query = st.text_input("🔍 Enter a keyword/phrase or Boolean query")
    k = st.slider("Top k matches", 1, 20, 5)
    if query:
        results = vectordb.similarity_search(query, k=k)
        st.markdown(f"### 🔍 Top {len(results)} passages for: *{query}*")
        for i, doc in enumerate(results, 1):
            src = doc.metadata["source"]
            pg  = doc.metadata["page_number"]
            # snippet ~150 words
            text = re.sub(r"\s+", " ", doc.page_content)
            words = text.split()
            idx = next((j for j,w in enumerate(words) 
                        if re.search(re.escape(query), w, re.IGNORECASE)), len(words)//2)
            start, end = max(0, idx-75), min(len(words), idx+75)
            snippet = " ".join(words[start:end])
            snippet = snippet.replace(query, f"**{query}**")
            with st.expander(f"{i}. {src}  (p.{pg})"):
                st.write(snippet + " …")
                pdf_url = f"{src}#page={pg}"
                components.iframe(pdf_url, width=700, height=500)
