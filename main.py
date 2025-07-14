# main.py
# Import necessary libraries
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

# --- Configuration ---
# Load environment variables for API keys and database connection
# Make sure to set these in your Vercel project settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="FastAPI RAG with Groq and Neon",
    description="A persistent RAG application using FastAPI, Groq, and Neon DB.",
    version="0.2.0",
)

# --- Global Variables & Initialization ---
# We will initialize these on application startup
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

    # Validate that environment variables are set
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable not set.")

    # Initialize the Groq Chat model
    chat = ChatGroq(temperature=0, model_name="Llama3-8b-8192", api_key=GROQ_API_KEY)

    # Initialize open-source embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Define the collection name for the vector store
    collection_name = "rag_documents"

    # Initialize the PGVector store
    # This will connect to your Neon database and use the specified collection
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=DATABASE_URL,
        use_jsonb=True, # Recommended for storing metadata
    )

    # Define the RAG prompt template
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
    """A simple root endpoint to confirm the server is running."""
    return {"message": "Welcome to the FastAPI RAG application with Groq and Neon DB!"}

@app.post("/upload/", summary="Upload and Process a Document")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document (.pdf, .txt, .docx). The file is processed,
    split into chunks, embedded, and stored in the persistent Neon vector store.
    """
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not initialized. Check server logs.")

    # Check for supported file types
    allowed_content_types = [
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Please upload a PDF, TXT, or DOCX file.")

    try:
        # Use a temporary file to handle the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Load the document using UnstructuredFileLoader
        loader = UnstructuredFileLoader(tmp_file_path)
        docs = loader.load()

        # Clean up the temporary file
        os.remove(tmp_file_path)

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Add the document chunks and their embeddings to the vector store
        await vectorstore.aadd_documents(splits)

        return {"status": "success", "message": f"Successfully processed and stored '{file.filename}'."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/query/", summary="Query the RAG Chain")
async def query_rag(query: str = Form(...)):
    """
    Endpoint to ask a question. It retrieves context from the Neon vector store
    and generates an answer, which is streamed back.
    """
    if not all([chat, prompt, vectorstore]):
        raise HTTPException(status_code=503, detail="RAG components not initialized. Check server logs.")

    # Create the retrieval chain on-the-fly for each query
    retriever = vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(chat, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    async def stream_response():
        """Generator function to stream the response from the RAG chain."""
        try:
            async for chunk in rag_chain.astream({"input": query}):
                if "answer" in chunk:
                    yield chunk["answer"]
        except Exception as e:
            error_message = f"Error during streaming: {str(e)}"
            print(error_message) # Log for debugging
            yield f"\n\n[ERROR: {error_message}]"

    return StreamingResponse(stream_response(), media_type="text/plain")
