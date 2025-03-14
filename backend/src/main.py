from fastapi import FastAPI, File, UploadFile, Form, Query
import io
import PyPDF2
from fastapi.middleware.cors import CORSMiddleware
from src.chunking import chunk_by_sentence, chunk_by_paragraph, chunk_by_page, chunk_by_tokens
import os
import json
import numpy as np
from src.embedding import EmbeddingGenerator
from src.similarity_metrics import SimilarityCalculator
from src.utils import format_context_for_llm, generate_llm_response

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
    "embeddings": [],
    "similarity_metric": "",
}

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(...),
    token_size: int = Form(256),
    sentence_size: int = Form(1),
    paragraph_size: int = Form(1),
    page_size: int = Form(1),
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
            chunks = chunk_by_sentence(full_text, size=sentence_size)
        elif chunking_strategy == "paragraph":
            chunks = chunk_by_paragraph(full_text, size=paragraph_size)
        elif chunking_strategy == "page":
            chunks = chunk_by_page(full_text, pages, size=page_size)
        elif chunking_strategy == "tokens":
            chunks = chunk_by_tokens(full_text, token_size=token_size)

        embeddings = EmbeddingGenerator.get_embeddings(chunks, embedding_model)

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
            chunk_embeddings = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_embeddings.append({
                    "chunk": chunk,
                    "chunk_no": i+1,
                    "embeddings": embedding
                })
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
def query_deep_learning_model(query: str, num_chunks: int = 5):
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
        
        # Get top k most relevant chunks based on user input
        top_chunks = SimilarityCalculator.get_top_k_chunks(
            current_document["chunks"], 
            similarity_scores, 
            k=num_chunks
        )
        
        top_chunk_texts = [chunk for _, (chunk, _) in top_chunks]
        context = format_context_for_llm(top_chunk_texts)
        llm_response = generate_llm_response(query, context)
        
        # Create a detailed answer with retrieved chunks and LLM response
        answer = llm_response
        
        # Return chunks and their scores along with the answer
        chunks_data = [
            {
                "chunk_number": chunk_idx + 1,  # Add 1 since chunk_idx is 0-based
                "text": chunk,
                "relevance_score": float(score)  # Convert numpy float to Python float
            }
            for chunk_idx, (chunk, score) in top_chunks
        ]
            
        return {
            "answer": answer,
            "chunks": chunks_data
        }
    
    except Exception as e:
        return {"answer": f"Error processing your query: {str(e)}"}