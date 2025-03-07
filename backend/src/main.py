from fastapi import FastAPI, File, UploadFile, Form, Query
import io
import PyPDF2
from fastapi.middleware.cors import CORSMiddleware
from src.chunking import chunk_by_sentence, chunk_by_paragraph, chunk_by_page, chunk_by_tokens
import os
import json

app = FastAPI(root_path='/api')

# list of allowed origins
origins = [
    "http://localhost:5173",
    "http://vcm-45508.vm.duke.edu"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)

# Global variable to store the current document and its chunks. For visualization, will be easy to retrieve chunks and embeddings.
current_document = {
    "text": "",
    "pages": [],
    "chunks": [],
    "chunking_strategy": "",
    "embedding_model": "",
    "embeddings": []
}

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(...),
    token_size: int = Form(256),
    embedding_model: str = Form(...),
):
    """
    Upload a PDF file and chunk it according to the specified strategy
    """
    try:
        # Read the PDF file
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Extract text from all pages
        full_text = ""
        pages = []
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            pages.append(page_text)
            full_text += page_text + "\n\n"
        
        # Apply chunking strategy
        chunks = []
        if chunking_strategy == "sentence":
            chunks = chunk_by_sentence(full_text)
        elif chunking_strategy == "paragraph":
            chunks = chunk_by_paragraph(full_text)
        elif chunking_strategy == "page":
            chunks = pages
        elif chunking_strategy == "tokens":
            chunks = chunk_by_tokens(full_text, token_size=token_size)

        if embedding_model == "sentence-transformer":
            embeddings = []
            pass
        elif embedding_model == "bert":
            embeddings = []
            pass
        elif embedding_model == "roberta":
            embeddings = []
            pass
        elif embedding_model == "distilbert":
            embeddings = []
            pass
        elif embedding_model == "gpt2":
            embeddings = []
            pass
        elif embedding_model == "fasttext":
            embeddings = []
            pass
        elif embedding_model == "use":
            embeddings = []
            pass
        elif embedding_model == "t5":
            embeddings = []
            pass
        else:
            embeddings = []

        
        # Store the document and chunks
        current_document["text"] = full_text
        current_document["pages"] = pages
        current_document["chunks"] = chunks
        current_document["chunking_strategy"] = chunking_strategy
        current_document["embedding_model"] = embedding_model
        current_document["embeddings"] = []
        
        # Save to disk for persistence
        with open("data/current_document.json", "w") as f:
            doc_data = {
                "filename": file.filename,
                "chunking_strategy": chunking_strategy,
                "chunk_count": len(chunks),
                "token_size": token_size if chunking_strategy == "tokens" else None,
                "embedding_model": embedding_model
            }
            json.dump(doc_data, f)


        with open("data/chunk_embeddings.json", "w") as f:
            # dump chunk as well as its embeddings
            chunk_embeddings = [{"chunk": chunk, "embeddings": embeddings} for chunk in chunks]
            json.dump(chunk_embeddings, f)

        
        return {
            "message": f"PDF processed with {chunking_strategy} chunking strategy and {embedding_model} model. Created {len(chunks)} chunks.",
            "chunk_count": len(chunks)
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/mean")
def query_mean_model(query: str):
    """
    Query endpoint for the mean model
    """
    # Pass query to some function
    answer = f"Response to the mean query : {query}"
    return {"answer": answer}

@app.get("/traditional")
def query_traditional_model(query: str):
    """
    Query endpoint for the traditional model
    """
    answer = f"Response to the traditional query : {query}"
    return {"answer": answer}

@app.get("/deep-learning")
def query_deep_learning_model(query: str):
    """
    Query endpoint for the deep learning model
    """
    # Check if we have a document loaded
    if not current_document["chunks"]:
        return {"answer": "Please upload a document first."}
    
    # TO DO
    # Get embeddings for the query
    # Compare with embeddings of chunks
    # Retrieve relevant chunks
    # Make call to LLM and Format response
    
    chunk_count = len(current_document["chunks"])
    strategy = current_document["chunking_strategy"]
    
    answer = f"Query: '{query}' processed against {chunk_count} chunks using {strategy} chunking strategy.\n\n"
    answer += "This would normally return relevant chunks and analysis based on semantic similarity."
    
    return {"answer": answer}