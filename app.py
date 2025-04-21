import os
import re
import streamlit as st
import streamlit.components.v1 as components

from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ── 1. App Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Student RAG MVP", layout="wide")
st.title("📖 Research Assistant with Inline PDF Viewer")

# ── 2. OpenAI Key ───────────────────────────────────────────────────────────
# In Streamlit Cloud: set OPENAI_API_KEY under Settings → Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ── 3. PDF Uploader ──────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDF(s) from your reading list",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    # 3.1 Chunk & Embed Prep
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for pdf in uploaded_files:
        # Save the upload to disk so the browser can fetch it
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        # 3.2 Load pages manually with pypdf into LangChain Documents
        reader = PdfReader(pdf.name)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf.name, "page_number": i},
                )
            )

        # 3.3 Split pages into ~500-token chunks
        chunks = splitter.split_documents(pages)
        for c in chunks:
            docs.append(c)

    st.success(f"Embedded {len(docs)} chunks from {len(uploaded_files)} files")

    # ── 4. Build Vector Store ──────────────────────────────────────────────────
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=None
    )

    # ── 5. Search UI ───────────────────────────────────────────────────────────
    query = st.text_input("🔍 Enter a keyword/phrase or Boolean query")
    k = st.slider("Top k matches", 1, 20, 5)

    if query:
        results = vectordb.similarity_search(query, k=k)
        st.markdown(f"### 🔍 Top {len(results)} passages for: *{query}*")
        for idx, doc in enumerate(results, start=1):
            src = doc.metadata["source"]
            pg  = doc.metadata["page_number"]
            # build a ~150‑word snippet around the first match
            text = re.sub(r"\s+", " ", doc.page_content)
            words = text.split()
            # find first occurrence
            pos = next((i for i,w in enumerate(words)
                        if re.search(re.escape(query), w, re.IGNORECASE)), len(words)//2)
            start, end = max(0, pos-75), min(len(words), pos+75)
            snippet = " ".join(words[start:end])
            snippet = re.sub(
                re.escape(query),
                lambda m: f"**{m.group(0)}**",
                snippet,
                flags=re.IGNORECASE
            )

            # Show each hit in an expander with inline PDF iframe
            with st.expander(f"{idx}. {src}  (p.{pg})"):
                st.write(snippet + " …")
                pdf_url = f"{src}#page={pg}"
                components.iframe(pdf_url, width=700, height=500)
