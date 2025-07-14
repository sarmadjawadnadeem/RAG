# main.py
# Import necessary libraries
import os
import tempfile
import httpx
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
# Correctly import the error class for the installed vercel-blob version
from vercel_blob import put, BlobUploadError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# Import the new Jina Embeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

# --- Configuration ---
# Load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
BLOB_READ_WRITE_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN")
# Add the new Jina API Key environment variable
JINA_API_KEY = os.environ.get("JINA_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FastAPI RAG with Jina Embeddings",
    description="A RAG application optimized for Vercel by using API-based embeddings.",
    version="0.5.2", # Incremented version for the fix
)

# --- Pydantic Models for Request Bodies ---
class UploadRequest(BaseModel):
    filename: str

class ProcessRequest(BaseModel):
    url: str
    filename: str

# --- Global Variables & Initialization ---
chat = None
embeddings = None
vectorstore = None
prompt = None

@app.on_event("startup")
def startup_event():
    """Initialize models and database connection on application startup."""
    global chat, embeddings, vectorstore, prompt
    try:
        print("INFO: Starting application initialization...")
        if not all([GROQ_API_KEY, DATABASE_URL, BLOB_READ_WRITE_TOKEN, JINA_API_KEY]):
            raise RuntimeError("One or more environment variables are not set. Please check all API keys and URLs.")
        
        chat = ChatGroq(temperature=0, model_name="Llama3-8b-8192", api_key=GROQ_API_KEY)
        embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY, model_name="jina-embeddings-v2-base-en")
        collection_name = "rag_documents"
        vectorstore = PGVector(embeddings=embeddings, collection_name=collection_name, connection=DATABASE_URL, use_jsonb=True)
        vectorstore.client.execute("SELECT 1")
        prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context. Think step-by-step. If the answer is not in the context, say so.
            <context>{context}</context>
            Question: {input}"""
        )
        print("INFO: Application initialization successful.")
    except Exception as e:
        print(f"FATAL: Application failed to start: {e}")
        raise

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint")
async def root():
    return {"message": "Welcome to the FastAPI RAG application with Jina Embeddings!"}

@app.post("/create-upload-url/", summary="Create a Pre-signed URL for Upload")
async def create_upload_url(request: UploadRequest):
    """
    STEP 1: The client requests a secure URL to upload a file directly to Vercel Blob.
    """
    try:
        blob = put(pathname=request.filename, body=None, add_random_suffix=True)
        return JSONResponse(content={"url": blob.url, "downloadUrl": blob.download_url})
    # Use the correct error class name based on the library's update
    except BlobUploadError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create upload URL: {str(e)}")

@app.post("/process-document/", summary="Process a Document from Vercel Blob")
async def process_document(request: ProcessRequest):
    """
    STEP 2: After the client uploads the file to the blob, it calls this endpoint
    with the blob's URL. The backend then downloads and processes the file.
    """
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized.")

    file_extension = os.path.splitext(request.filename)[1].lower()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.url)
            response.raise_for_status()
            file_content = response.content

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        await vectorstore.aadd_documents(splits)

        return {"status": "success", "message": f"Successfully processed '{request.filename}'."}
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/query/", summary="Query the RAG Chain")
async def query_rag(query: str = Form(...)):
    if not all([chat, prompt, vectorstore]):
        raise HTTPException(status_code=503, detail="RAG components not initialized.")

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
