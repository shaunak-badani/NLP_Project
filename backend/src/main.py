from fastapi import FastAPI, File, UploadFile, Form
import io
import PyPDF2
from fastapi.middleware.cors import CORSMiddleware
from chunking import chunk_by_sentence, chunk_by_paragraph, chunk_by_tokens
import os
import json
import sys
import numpy as np
import tempfile
from embedding import EmbeddingGenerator
from similarity_metrics import SimilarityCalculator
from utils import format_context_for_llm, generate_llm_response
from Naive import ChunkedTextSearcher

app = FastAPI(root_path='/api')

# list of allowed origins
origins = [
    "http://localhost:5173",
    "http://vcm-45508.vm.duke.edu"
    "http://localhost:5174"
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
    "embeddings": [],
    "similarity_metric": "",
}

# Global variable for the naive searcher
naive_searcher = None

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(...),
    token_size: int = Form(256),
    embedding_model: str = Form(...),
    similarity_metric: str = Form(...),
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
        elif chunking_strategy == "tokens":
            chunks = chunk_by_tokens(full_text, token_size=token_size)

        embeddings = EmbeddingGenerator.get_embeddings(chunks, embedding_model)

        print(len(chunks))
        print(len(embeddings))

        # Store the document and chunks
        current_document["text"] = full_text
        current_document["pages"] = pages
        current_document["chunks"] = chunks
        current_document["chunking_strategy"] = chunking_strategy
        current_document["embedding_model"] = embedding_model
        current_document["embeddings"] = embeddings
        current_document["similarity_metric"] = similarity_metric
        
        # Save to disk for persistence
        with open("data/current_document.json", "w") as f:
            doc_data = {
                "filename": file.filename,
                "chunking_strategy": chunking_strategy,
                "chunk_count": len(chunks),
                "token_size": token_size if chunking_strategy == "tokens" else None,
                "embedding_model": embedding_model,
                "similarity_metric": similarity_metric,
            }
            json.dump(doc_data, f)


        with open("data/chunk_embeddings.json", "w") as f:
            # dump chunk as well as its embeddings
            chunk_embeddings = [{"chunk": chunk, "embeddings": embedding} for chunk, embedding in zip(chunks, embeddings)]
            json.dump(chunk_embeddings, f)

        
        return {
            "message": f"PDF processed with {chunking_strategy} chunking strategy and {embedding_model} model. Created {len(chunks)} chunks.",
            "chunk_count": len(chunks)
        }
    
    except Exception as e:
        return {"message": str(e)}

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
    
    try:
        # Get embeddings for the query using the same model as the chunks
        query_embedding = EmbeddingGenerator.get_embeddings([query], current_document["embedding_model"])[0]
        
        # Calculate similarity scores based on selected metric
        similarity_scores = SimilarityCalculator.get_similarity_scores(
            query_embedding, 
            current_document["embeddings"], 
            current_document["similarity_metric"]
        )
        
        # Get top 5 most relevant chunks
        top_chunks = SimilarityCalculator.get_top_k_chunks(
            current_document["chunks"], 
            similarity_scores, 
            k=5
        )
        
        top_chunk_texts = [chunk for chunk, score in top_chunks]
        context = format_context_for_llm(top_chunk_texts)
        llm_response = generate_llm_response(query, context)
        
        # Create a detailed answer with retrieved chunks and LLM response
        answer = llm_response
        
        # answer += "--- Top Relevant Sections ---\n"
        # for i, (chunk, score) in enumerate(top_chunks, 1):
        #     # Truncate very long chunks for display
        #     display_chunk = chunk[:150] + "..." if len(chunk) > 150 else chunk
        #     answer += f"{i}. Relevance: {score:.4f}\n{display_chunk}\n\n"
            
        return {"answer": answer}
    
    except Exception as e:
        return {"answer": f"Error processing your query: {str(e)}"}

# New endpoints for the Naive approach - FIXED ROUTES

@app.post("/upload-naive")
async def upload_naive_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(...),
    token_size: int = Form(256),
    overlap: int = Form(20),
    similarity_metric: str = Form("cosine"),
):
    """
    Upload a file and process it with the Naive approach
    """
    print(f"Received upload request: chunking_strategy={chunking_strategy}, token_size={token_size}")
    try:
        global naive_searcher
        
        # Create a new searcher with specified parameters
        naive_searcher = ChunkedTextSearcher(
            chunking_method=chunking_strategy,
            chunk_size=token_size,
            overlap=overlap
        )
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())
            print(f"Temp file created: {tmp_path}")
        
        # Add the file to the searcher
        file_name = file.filename or "uploaded_document"
        print(f"Processing file: {file_name}, type: {file.content_type}")
        
        if file_name.lower().endswith('.pdf'):
            success = naive_searcher.add_pdf(file_name, tmp_path)
        else:
            # For non-PDF files, read as text
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            success = naive_searcher.add_text(file_name, content)
        
        # Clean up the temp file
        os.unlink(tmp_path)
        
        if not success:
            return {"message": f"Failed to process {file_name}"}
        
        # Build the search index
        naive_searcher.build_index()
        
        return {
            "message": f"Document processed with {chunking_strategy} chunking strategy. Created {len(naive_searcher.chunks.get(file_name, []))} chunks."
        }
    
    except Exception as e:
        print(f"Error in upload_naive_document: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"message": f"Error: {str(e)}"}

@app.get("/mean-naive")
def query_naive_model(query: str, num_results: int = 3, similarity_metric: str = "cosine"):
    """
    Query endpoint for the naive model using TF-IDF
    """
    global naive_searcher
    
    if not naive_searcher:
        return {"answer": "Please upload a document first."}
    
    try:
        # Search using the naive searcher
        results = naive_searcher.search(
            query=query,
            num_results=num_results,
            similarity_metric=similarity_metric
        )
        
        # Format the response
        if not results:
            answer = "No relevant results found for your query."
        else:
            # Generate a summary of the results
            answer = f"Here are the top {len(results)} results for your query:\n\n"
            
            for i, result in enumerate(results, 1):
                answer += f"{i}. {result['snippet']}\n\n"
        
        return {
            "answer": answer,
            "results": results  # Return the raw results for UI rendering
        }
    
    except Exception as e:
        return {"answer": f"Error processing your query: {str(e)}"}