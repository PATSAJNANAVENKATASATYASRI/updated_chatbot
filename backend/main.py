import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Any

# LangChain imports
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Core for custom retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as CoreDocument
from langchain_core.pydantic_v1 import Field

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file.")

# --- Paths ---
PDF_VECTOR_DB_PATH = "chroma_db_pdfs/"
WEB_VECTOR_DB_PATH = "chroma_db_web/"

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

# --- Globals ---
conversation_chain = None
pdf_vectorstore = None
web_vectorstore = None

# --- Custom Combined Retriever ---
class CombinedRetriever(BaseRetriever):
    retrievers: List[BaseRetriever] = Field(...)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[CoreDocument]:
        docs = []
        for retriever in self.retrievers:
            try:
                docs.extend(retriever.get_relevant_documents(query, **kwargs))
            except Exception as e:
                print(f"⚠️ Sync retriever failed: {e}")
        return self._deduplicate(docs)

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[CoreDocument]:
        docs = []
        for retriever in self.retrievers:
            try:
                result = await retriever.ainvoke(query, **kwargs)
                docs.extend(result)
            except Exception as e:
                print(f"⚠️ Async retriever failed: {e}")
        return self._deduplicate(docs)

    def _deduplicate(self, docs: List[CoreDocument]) -> List[CoreDocument]:
        seen = set()
        unique = []
        for doc in docs:
            doc_identifier = (doc.page_content, doc.metadata.get('source'))
            if doc_identifier not in seen:
                unique.append(doc)
                seen.add(doc_identifier)
        return unique

# --- Startup: Load vectorstores and initialize chain ---
@app.on_event("startup")
async def startup_event():
    global conversation_chain, pdf_vectorstore, web_vectorstore

    print("🔧 Initializing RAG system...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(PDF_VECTOR_DB_PATH) and os.listdir(PDF_VECTOR_DB_PATH):
        pdf_vectorstore = Chroma(persist_directory=PDF_VECTOR_DB_PATH, embedding_function=embeddings)
        print("✅ PDF vectorstore loaded.")
    else:
        print("⚠️ PDF vectorstore not found. Run ingest.py first.")

    if os.path.exists(WEB_VECTOR_DB_PATH) and os.listdir(WEB_VECTOR_DB_PATH):
        web_vectorstore = Chroma(persist_directory=WEB_VECTOR_DB_PATH, embedding_function=embeddings)
        print("✅ Web vectorstore loaded.")
    else:
        print("⚠️ Web vectorstore not found. Run ingest_web_only.py first.")

    # ✅ Use valid Gemini model name
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, api_key=GOOGLE_API_KEY)
    print("🧠 Gemini LLM ready.")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retrievers = []
    if pdf_vectorstore:
        retrievers.append(pdf_vectorstore.as_retriever(search_kwargs={"k": 5}))
    if web_vectorstore:
        retrievers.append(web_vectorstore.as_retriever(search_kwargs={"k": 5}))

    if retrievers:
        combined_retriever = CombinedRetriever(retrievers=retrievers)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=combined_retriever,
            memory=memory,
            return_source_documents=True
        )
        print("✅ RAG chain initialized with PDFs + Web data.")
    else:
        print("⚠️ No retrievers available. Chat will not work until vectorstores are loaded.")

# --- Chat endpoint ---
@app.post("/chat")
async def chat_with_assistant(request: ChatRequest):
    if not conversation_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    query = request.query.strip()
    if not query:
        return {"response": "Please enter a query."}

    if re.search(r"\b(how do i go|get to|directions|route|navigate|way to)\b", query.lower()):
        return {"response": "Navigation feature is under development."}

    try:
        result = await conversation_chain.ainvoke({"question": query})
        sources = result.get("source_documents", [])
        print(f"🔍 Retrieved {len(sources)} documents:")
        for i, doc in enumerate(sources):
            print(f"\n📄 Source {i+1}: {doc.metadata.get('source', 'N/A')}")
            print(doc.page_content[:300] + "...")

        response_text = result["answer"]
        source_links = []

        for doc in sources:
            source = doc.metadata.get("source")
            url = doc.metadata.get("url")
            if source and url:
                source_links.append(f"📄 Found in <a href='{url}' target='_blank'>{source}</a>")

        if source_links:
            response_text += "<br><br>" + "<br>".join(source_links)

        return {"response": response_text}

    except Exception as e:
        print(f"❌ Error during chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
