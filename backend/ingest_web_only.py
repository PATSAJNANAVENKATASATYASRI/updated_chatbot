# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from datetime import datetime

# # --- Config ---
# SEED_URL = "https://srivasaviengg.ac.in/"
# MAX_PAGES = 100
# DB_PATH = "chroma_db_college/"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50

# # --- Logging ---
# def log(msg):
#     print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# # --- Helpers ---
# def is_internal(url, base):
#     return urlparse(url).netloc == urlparse(base).netloc

# def is_pdf(url):
#     return url.lower().endswith(".pdf")

# def clean_html(html):
#     soup = BeautifulSoup(html, "html.parser")
#     for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
#         tag.decompose()
#     return " ".join(soup.get_text(separator=" ").split())

# def extract_pdf_title(url):
#     return url.split("/")[-1].replace("-", " ").replace("_", " ").replace(".pdf", "").strip()

# def fetch_with_retry(url, retries=2, delay=5):
#     headers = {"User-Agent": "Mozilla/5.0"}
#     for i in range(retries):
#         try:
#             safe_url = requests.utils.requote_uri(url)
#             r = requests.get(safe_url, timeout=15, headers=headers)
#             if r.status_code == 200:
#                 return r
#         except Exception as e:
#             log(f"Retry {i+1} for {url} due to {e}")
#     return None

# def get_internal_links(html, base_url):
#     soup = BeautifulSoup(html, "html.parser")
#     links = set()
#     for tag in soup.find_all("a", href=True):
#         href = tag['href']
#         full_url = urljoin(base_url, href)
#         if is_internal(full_url, base_url):
#             links.add(full_url.split("#")[0])
#     return links

# # --- Main ---
# def crawl_and_ingest():
#     visited = set()
#     to_visit = [SEED_URL]
#     docs = []

#     while to_visit and len(visited) < MAX_PAGES:
#         url = to_visit.pop()
#         if url in visited:
#             continue

#         log(f"🌐 Crawling: {url}")
#         response = fetch_with_retry(url)
#         if not response:
#             log(f"⚠️ Failed to crawl {url}")
#             continue

#         visited.add(url)
#         content_type = response.headers.get("Content-Type", "")

#         if "text/html" in content_type:
#             html = response.text
#             text = clean_html(html)
#             if text.strip():
#                 docs.append(Document(page_content=text, metadata={"source": url}))
#             new_links = get_internal_links(html, SEED_URL)
#             to_visit.extend(new_links - visited)

#         elif is_pdf(url):
#             title = extract_pdf_title(url)
#             docs.append(Document(page_content=title, metadata={"source": url}))

#     if not docs:
#         log("❌ No documents to ingest.")
#         return

#     log(f"📄 Extracted {len(docs)} documents. Chunking...")
#     splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#     chunks = splitter.split_documents(docs)

#     log(f"🧠 Embedding {len(chunks)} chunks...")
#     embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
#     store.add_documents(chunks)
#     store.persist()

#     log(f"✅ Ingested {len(chunks)} chunks into Chroma DB.")

# if __name__ == "__main__":
#     crawl_and_ingest()
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SEED_URL     = "https://srivasaviengg.ac.in/"
MAX_PAGES    = 100
DB_PATH      = "chroma_db_college"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def is_internal(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc

def is_pdf(url: str) -> bool:
    return url.lower().endswith(".pdf")

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())

def extract_pdf_title(url: str) -> str:
    return url.split("/")[-1].replace("-", " ").replace("_", " ").replace(".pdf", "").strip()

def fetch_with_retry(url: str, retries: int = 2, delay: int = 5):
    headers = {"User-Agent": "Mozilla/5.0"}
    for i in range(retries):
        try:
            safe_url = requests.utils.requote_uri(url)
            r = requests.get(safe_url, timeout=15, headers=headers)
            if r.status_code == 200:
                return r
        except Exception as e:
            log(f"Retry {i+1} for {url} due to {e}")
    return None

def get_internal_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        full_url = urljoin(base_url, href)
        if is_internal(full_url, base_url):
            links.add(full_url.split("#")[0])
    return links


def crawl_and_ingest():
    visited, to_visit, docs = set(), [SEED_URL], []

    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop()
        if url in visited:
            continue

        log(f"🌐 Crawling: {url}")
        response = fetch_with_retry(url)
        if not response:
            log(f"⚠️ Failed to crawl {url}")
            continue

        visited.add(url)
        ctype = response.headers.get("Content-Type", "")

        if "text/html" in ctype:
            html = response.text
            text = clean_html(html)
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": url}))
            new_links = get_internal_links(html, SEED_URL)
            to_visit.extend(new_links - visited)

        elif is_pdf(url):
            title = extract_pdf_title(url)
            docs.append(Document(page_content=title, metadata={"source": url}))

    if not docs:
        log("❌ No documents to ingest.")
        return

    log(f"📄 Extracted {len(docs)} documents. Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                              chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    log(f"🧠 Embedding {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    store.add_documents(chunks)
    store.persist()
    log(f"✅ Ingested {len(chunks)} chunks into Chroma DB at '{DB_PATH}'.")


if __name__ == "__main__":
    crawl_and_ingest()
