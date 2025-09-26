"""
ingest.py
Robust ingestion of a very large set of PDFs into a Chroma vector store.

Requirements:
    pip install langchain langchain-community langchain-chroma
    pip install sentence-transformers pypdf
"""

import os
import glob
import gc
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain.schema import Document


# ---------------------- CONFIG ----------------------
PDF_DIR        = "pdfs"            # Folder containing PDFs
CHROMA_DIR     = "chroma_db_pdfs"  # Vector DB location
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_FILE_CT  = 50      # PDFs per batch
CHUNK_SIZE     = 1200    # characters per chunk
CHUNK_OVERLAP  = 150     # overlap between chunks
MAX_CHUNK_LEN  = 4000    # absolute hard cap for a chunk
# -----------------------------------------------------


def find_good_pdfs(pdf_dir: str) -> List[str]:
    """Return only valid, readable PDF paths."""
    pdf_paths = glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True)
    good = []
    for p in pdf_paths:
        try:
            PdfReader(p)  # quick validation
            good.append(p)
        except Exception as e:
            print(f"⚠️  Skipping corrupt PDF: {p} ({e})")
    return good


def load_and_split(pdf_paths: List[str]) -> List[Document]:
    """Load PDFs and split into clean text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    documents: List[Document] = []
    for path in pdf_paths:
        print(f"📄 Loading {path}")
        loader = PyPDFLoader(path)
        for doc in loader.load():
            text = doc.page_content.strip()
            if not text:
                continue
            # truncate extreme length
            if len(text) > MAX_CHUNK_LEN:
                text = text[:MAX_CHUNK_LEN]
            for chunk in splitter.split_text(text):
                documents.append(Document(page_content=chunk,
                                          metadata={"source": os.path.basename(path)}))
    return documents


def build_vectorstore(all_pdf_paths: List[str]):
    """
    Embed documents in batches and write to Chroma incrementally.
    Uses the new save_local() method instead of deprecated persist().
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Create (or load) a persistent Chroma store
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    for i in range(0, len(all_pdf_paths), BATCH_FILE_CT):
        batch_paths = all_pdf_paths[i: i + BATCH_FILE_CT]
        print(f"\n🚀 Processing batch {i // BATCH_FILE_CT + 1} "
              f"({len(batch_paths)} PDFs)")

        docs = load_and_split(batch_paths)
        print(f"   ➡️  {len(docs)} text chunks to embed")

        if docs:
            vs.add_documents(docs)
            # NEW: save_local replaces the old persist()
            # vs.persist()
            print("   ✅ Batch committed to Chroma")

        # free memory between batches
        del docs
        gc.collect()

    print("\n🎯 Ingestion complete.")


if __name__ == "__main__":
    print("🔍 Scanning PDFs…")
    valid_pdfs = find_good_pdfs(PDF_DIR)
    print(f"✅ {len(valid_pdfs)} valid PDFs found.")
    if not valid_pdfs:
        raise SystemExit("No valid PDFs to ingest.")

    build_vectorstore(valid_pdfs)
"""
ingest.py
Robust ingestion of a large set of PDFs into a Chroma vector store.

Requirements:
    pip install langchain langchain-community langchain-chroma
    pip install sentence-transformers pypdf
"""

# import os
# import glob
# import gc
# import warnings
# from typing import List

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from pypdf import PdfReader
# from langchain.schema import Document

# # Suppress future warnings from transformers
# warnings.filterwarnings("ignore", category=FutureWarning)

# # ---------------------- CONFIG ----------------------
# PDF_DIR        = "pdfs"            # Folder containing PDFs
# CHROMA_DIR     = "chroma_db_pdfs"  # Vector DB location
# EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

# BATCH_FILE_CT  = 50      # PDFs per batch
# CHUNK_SIZE     = 1200    # characters per chunk
# CHUNK_OVERLAP  = 150     # overlap between chunks
# MAX_CHUNK_LEN  = 4000    # absolute hard cap for a chunk
# # -----------------------------------------------------


# def find_good_pdfs(pdf_dir: str) -> List[str]:
#     """Return only valid, readable PDF paths."""
#     pdf_paths = glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True)
#     good = []
#     for p in pdf_paths:
#         try:
#             PdfReader(p)  # quick validation
#             good.append(p)
#         except Exception as e:
#             print(f"⚠️  Skipping corrupt PDF: {p} ({e})")
#     return good


# def load_and_split(pdf_paths: List[str]) -> List[Document]:
#     """Load PDFs and split into clean text chunks."""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP
#     )

#     documents: List[Document] = []
#     for path in pdf_paths:
#         print(f"📄 Loading {path}")
#         loader = PyPDFLoader(path)
#         for doc in loader.load():
#             text = doc.page_content.strip()
#             if not text:
#                 continue
#             if len(text) > MAX_CHUNK_LEN:
#                 text = text[:MAX_CHUNK_LEN]
#             for chunk in splitter.split_text(text):
#                 if chunk.strip():  # skip empty chunks
#                     documents.append(Document(
#                         page_content=chunk,
#                         metadata={"source": os.path.basename(path)}
#                     ))
#     return documents


# def build_vectorstore(all_pdf_paths: List[str]):
#     """Embed documents in batches and write to Chroma incrementally."""
#     os.makedirs(CHROMA_DIR, exist_ok=True)

#     embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

#     # Create (or load) a persistent Chroma store
#     vs = Chroma(
#         persist_directory=CHROMA_DIR,
#         embedding_function=embeddings
#     )

#     for i in range(0, len(all_pdf_paths), BATCH_FILE_CT):
#         batch_paths = all_pdf_paths[i: i + BATCH_FILE_CT]
#         print(f"\n🚀 Processing batch {i // BATCH_FILE_CT + 1} "
#               f"({len(batch_paths)} PDFs)")

#         docs = load_and_split(batch_paths)
#         print(f"   ➡️  {len(docs)} text chunks to embed")

#         # Sanitize chunks before embedding
#         docs = [doc for doc in docs if isinstance(doc.page_content, str) and doc.page_content.strip()]

#         if docs:
#             try:
#                 vs.add_documents(docs)
#                 print("   ✅ Batch committed to Chroma")
#             except Exception as e:
#                 print(f"❌ Failed to embed batch: {e}")

#         # Free memory between batches
#         del docs
#         gc.collect()

#     print("\n🎯 Ingestion complete.")


# if __name__ == "__main__":
#     print("🔍 Scanning PDFs…")
#     valid_pdfs = find_good_pdfs(PDF_DIR)
#     print(f"✅ {len(valid_pdfs)} valid PDFs found.")
#     if not valid_pdfs:
#         raise SystemExit("No valid PDFs to ingest.")

#     build_vectorstore(valid_pdfs)