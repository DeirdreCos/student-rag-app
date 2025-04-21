import os
import re
import io
import base64

import streamlit as st
import streamlit.components.v1 as components

from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# ── 1. App Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Student RAG MVP", layout="wide")
st.title("📖 Research Assistant with 600‑Word Page Previews")

# ── 2. Load API Key & Session State ──────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

# ── 3. Upload PDFs & Build both Vector Store + Full‑Page Texts ───────────────
uploaded = st.file_uploader(
    "Upload PDF(s) from your reading list",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded:
    # 3a) Read into memory, build a base64 data‑URL (we won’t embed it, but it’s here)
    pdf_data = {}
    for pdf in uploaded:
        raw = pdf.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        pdf_data[pdf.name] = "data:application/pdf;base64," + b64
        pdf.seek(0)

    # 3b) Extract full text **per page** for later preview
    page_texts = {}
    pages = []
    for pdf in uploaded:
        reader = PdfReader(io.BytesIO(pdf.read()))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            page_texts[(pdf.name, i)] = text
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf.name, "page_number": i},
                )
            )
        pdf.seek(0)

    # 3c) Chunk each page for RAG
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    st.success(f"🔗 Embedded {len(docs)} chunks from {len(uploaded)} file(s)")

    # ── 4. Build Vector Store ──────────────────────────────────────────────────
    embeddings = OpenAIEmbeddings()
    vectordb    = FAISS.from_documents(documents=docs, embedding=embeddings)

    # ── 5. Search UI & Two‑Column Results ──────────────────────────────────────
    query = st.text_input("🔍 Enter a keyword/phrase or Boolean query")
    k     = st.slider("Top k matches", 1, 20, 5)

    if query:
        results = vectordb.similarity_search(query, k=k)
        st.markdown(f"### 🔍 Top {len(results)} passages for *{query}*")

        left, right = st.columns([2, 3])

        # LEFT column: same expandable snippets + “View full page” buttons
        with left:
            for idx, doc in enumerate(results, start=1):
                src = doc.metadata["source"]
                pg  = doc.metadata["page_number"]

                # build ~150‑word snippet around the query
                text   = re.sub(r"\s+", " ", doc.page_content)
                words  = text.split()
                pos    = next(
                    (i for i, w in enumerate(words)
                     if re.search(re.escape(query), w, re.IGNORECASE)),
                    len(words)//2,
                )
                start, end = max(0, pos-75), min(len(words), pos+75)
                snippet    = " ".join(words[start:end])
                snippet    = re.sub(
                    re.escape(query),
                    lambda m: f"**{m.group(0)}**",
                    snippet,
                    flags=re.IGNORECASE,
                )

                with st.expander(f"{idx}. {src}  (p.{pg})"):
                    st.write(snippet + " …")
                    if st.button("View full page", key=f"view_{idx}"):
                        st.session_state.selected_pdf = (src, pg)

        # RIGHT column: show a 600‑word preview of the **entire page**
        with right:
            if st.session_state.selected_pdf:
                fname, page = st.session_state.selected_pdf
                full_text   = page_texts[(fname, page)]
                words       = full_text.split()
                preview     = " ".join(words[:600])
                if len(words) > 600:
                    preview += " …"

                st.markdown(f"**Preview of page {page} from {fname}:**")
                st.write(preview)
            else:
                st.info("Click **View full page** on the left to load a page preview here.")
