# main.py
# Import necessary libraries
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import specific, lightweight document loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

# --- Configuration ---
# Load environment variables for API keys and database connection
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FastAPI RAG with Groq and Neon (Memory Optimized)",
    description="A persistent RAG application using lightweight loaders to deploy on Vercel.",
    version="0.3.0",
)

# --- Global Variables & Initialization ---
chat = None
embeddings = None
vectorstore = None
prompt = None

@app.on_event("startup")
def startup_event():
    """
    Initialize models and database connection on application startup.
    """
    global chat, embeddings, vectorstore, prompt

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable not set.")

    chat = ChatGroq(temperature=0, model_name="Llama3-8b-8192", api_key=GROQ_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = "rag_documents"
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context.
        Think step-by-step and provide the best possible answer.
        If the answer is not in the context, say, "I can't answer this question as it is not in the provided context."

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint")
async def root():
    return {"message": "Welcome to the FastAPI RAG application with Groq and Neon DB!"}

@app.post("/upload/", summary="Upload and Process a Document")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document (.pdf, .txt, .docx). The file is processed
    using lightweight loaders to avoid memory issues on Vercel.
    """
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Select the appropriate loader based on the file extension
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(tmp_file_path)
        else:
            os.remove(tmp_file_path)
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, TXT, or DOCX file.")

        docs = loader.load()
        os.remove(tmp_file_path) # Clean up the temporary file

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        await vectorstore.aadd_documents(splits)

        return {"status": "success", "message": f"Successfully processed and stored '{file.filename}'."}

    except Exception as e:
        # Ensure temp file is removed on error if it exists
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
            error_message = f"Error during streaming: {str(e)}"
            print(error_message)
            yield f"\n\n[ERROR: {error_message}]"

    return StreamingResponse(stream_response(), media_type="text/plain")
