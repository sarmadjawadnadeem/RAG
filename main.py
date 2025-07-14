# main.py
# Import necessary libraries
import os
import tempfile
import httpx
import traceback
from functools import lru_cache
from fastapi import FastAPI, Form, HTTPException, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector
from vercel_blob import put

# --- Configuration ---
# Load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
BLOB_READ_WRITE_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN")
JINA_API_KEY = os.environ.get("JINA_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FastAPI RAG with Jina Embeddings",
    description="A RAG application optimized for Vercel with robust dependency injection.",
    version="0.7.1", # Added enhanced logging
)

# --- Dependency Injection Setup ---
# Using lru_cache to ensure components are initialized only once per instance (on warm starts)

@lru_cache(maxsize=1)
def get_chat_model():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    return ChatGroq(temperature=0, model_name="Llama3-8b-8192", api_key=GROQ_API_KEY)

@lru_cache(maxsize=1)
def get_embeddings():
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY not set")
    return JinaEmbeddings(jina_api_key=JINA_API_KEY, model_name="jina-embeddings-v2-base-en")

@lru_cache(maxsize=1)
def get_vectorstore():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    embeddings = get_embeddings()
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="rag_documents",
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    # Test connection
    vectorstore.client.execute("SELECT 1")
    return vectorstore

def get_prompt_template():
    return ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context.
        Think step-by-step. If the answer is not in the context, say so.
        <context>{context}</context>
        Question: {input}"""
    )

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint")
async def root():
    return {"message": "Welcome to the FastAPI RAG application with Jina Embeddings!"}

@app.post("/upload-and-process/", summary="Upload and Process a Document")
async def upload_and_process(
    file: UploadFile = File(...),
    vectorstore: PGVector = Depends(get_vectorstore)
):
    """
    This single endpoint handles uploading a file and processing it into the vector store.
    """
    try:
        print("INFO: Starting upload and process.")
        file_content = await file.read()
        print(f"INFO: Read {len(file_content)} bytes from file '{file.filename}'.")

        # Upload to Vercel Blob
        print("INFO: Uploading to Vercel Blob...")
        blob = put(file.filename, file_content)
        print(f"INFO: Successfully uploaded to Vercel Blob. URL: {blob.url}")

        # Process the file
        print("INFO: Writing to temporary file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        print(f"INFO: Temporary file created at {tmp_file_path}")

        file_extension = os.path.splitext(file.filename)[1].lower()
        print(f"INFO: Loading document with extension {file_extension}...")
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(tmp_file_path)
        else:
            os.remove(tmp_file_path)
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        docs = loader.load()
        os.remove(tmp_file_path)
        print(f"INFO: Document loaded. Number of pages/docs: {len(docs)}")

        print("INFO: Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"INFO: Document split into {len(splits)} chunks.")
        
        # Add documents to the vector store
        print("INFO: Adding documents to vector store...")
        await vectorstore.aadd_documents(splits)
        print("INFO: Successfully added documents to vector store.")

        return {"status": "success", "filename": file.filename, "blob_url": blob.url}
    except Exception as e:
        print(f"ERROR during upload/processing: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/query/", summary="Query the RAG Chain")
async def query_rag(
    query: str = Form(...),
    chat: ChatGroq = Depends(get_chat_model),
    vectorstore: PGVector = Depends(get_vectorstore),
    prompt: ChatPromptTemplate = Depends(get_prompt_template)
):
    """
    This endpoint takes a query, retrieves context, and generates an answer.
    """
    retriever = vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(chat, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    async def stream_response():
        try:
            async for chunk in rag_chain.astream({"input": query}):
                if "answer" in chunk:
                    yield chunk["answer"]
        except Exception as e:
            yield f"\n\n[ERROR: {str(e)}]"

    return StreamingResponse(stream_response(), media_type="text/plain")
