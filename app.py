import os
import re
import base64
import io

import streamlit as st
import streamlit.components.v1 as components

from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# ── 1. App Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Student RAG MVP", layout="wide")
st.title("📖 Research Assistant with Inline PDF Viewer")

# ── 2. Load OpenAI Key & Viewer State ───────────────────────────────────────
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

# ── 3. Upload & In‑Memory PDF Encoding ───────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDF(s) from your reading list",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    # Turn each PDF into a base64 data‑URL
    pdf_data = {}
    for pdf in uploaded_files:
        raw = pdf.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        pdf_data[pdf.name] = "data:application/pdf;base64," + b64
        pdf.seek(0)

    # Split into pages → chunks
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for pdf in uploaded_files:
        reader = PdfReader(io.BytesIO(pdf.read()))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(Document(
                page_content=text,
                metadata={"source": pdf.name, "page_number": i}
            ))
        docs.extend(splitter.split_documents(pages))
        pdf.seek(0)

    st.success(f"🔗 Embedded {len(docs)} chunks from {len(uploaded_files)} file(s)")

    # ── 4. Build Vector Store ──────────────────────────────────────────────────
    embeddings = OpenAIEmbeddings()
    vectordb    = FAISS.from_documents(documents=docs, embedding=embeddings)

    # ── 5. Search & Two‑Column Viewer ──────────────────────────────────────────
    query = st.text_input("🔍 Enter a keyword/phrase or Boolean query")
    k     = st.slider("Top k matches", 1, 20, 5)

    if query:
        results = vectordb.similarity_search(query, k=k)
        st.markdown(f"### 🔍 Top {len(results)} passages for: *{query}*")

        left, right = st.columns([2, 3])

        # LEFT: snippets + “View full page” buttons
        with left:
            for idx, doc in enumerate(results, start=1):
                src = doc.metadata["source"]
                pg  = doc.metadata["page_number"]

                # build ~150‑word snippet around the query
                text  = re.sub(r"\s+", " ", doc.page_content)
                words = text.split()
                pos   = next(
                    (i for i,w in enumerate(words)
                     if re.search(re.escape(query), w, re.IGNORECASE)),
                    len(words)//2
                )
                start, end = max(0, pos-75), min(len(words), pos+75)
                snippet = " ".join(words[start:end])
                snippet = re.sub(
                    re.escape(query),
                    lambda m: f"**{m.group(0)}**",
                    snippet,
                    flags=re.IGNORECASE
                )

                with st.expander(f"{idx}. {src}  (p.{pg})"):
                    st.write(snippet + " …")
                    if st.button("View full page", key=f"view_{idx}"):
                        st.session_state.selected_pdf = (src, pg)

        # ── RIGHT: show PDF page + fallback link ────────────────────────────
        with right:
            if st.session_state.selected_pdf:
                fname, page = st.session_state.selected_pdf
                data_url = pdf_data[fname]

                # Try the iframe first
                iframe_html = f'''
                    <iframe
                        src="{data_url}#page={page}"
                        width="700"
                        height="800"
                    ></iframe>
                '''
                components.html(iframe_html, height=820)

                # Fallback link
                st.markdown(
                    f"🔗 [Open this PDF page in a new tab]({data_url}#page={page})",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Click **View full page** on the left to load the PDF here.")
