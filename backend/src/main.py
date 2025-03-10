from fastapi import FastAPI, File, UploadFile, Form, Query
import io
import PyPDF2
from fastapi.middleware.cors import CORSMiddleware
from chunking import chunk_by_sentence, chunk_by_paragraph, chunk_by_page, chunk_by_tokens
import os
import json
import numpy as np
from embedding import EmbeddingGenerator
from similarity_metrics import SimilarityCalculator
from utils import format_context_for_llm, generate_llm_response
from visualization import PCA_visualization, tSNE_visualization, UMAP_visualization
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fastapi.responses import JSONResponse

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

query_embedding = []
similarity_scores = [] 

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
        elif chunking_strategy == "page":
            chunks = pages
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

    global query_embedding, similarity_scores

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
    


@app.get("/visualize-embeddings")
async def visualize_embeddings(method: str = Query("pca"), k: int = Query(5)):
    global similarity_scores

    if not current_document["embeddings"]:
        return {"error": "No document uploaded yet."}

    if not query_embedding:
        return {"error": "No query embedding available. Send a query first."}

    if not similarity_scores:
        return {"error": "No similarity scores available. Run a query first."}

    try:
        chunks_embs = np.array(current_document["embeddings"])
        query_emb = np.array(query_embedding)

        top_chunk_indices = np.argsort(similarity_scores)[-k:] + 1  

        if method.lower() == "pca":
            img_base64 = PCA_visualization(chunks_embs, query_emb, top_chunk_indices)
        elif method.lower() == "tsne":
            img_base64 = tSNE_visualization(chunks_embs, query_emb, top_chunk_indices)
        elif method.lower() == "umap":
            img_base64 = UMAP_visualization(chunks_embs, query_emb, top_chunk_indices)
        else:
            return {"error": "Invalid visualization method"}

        return {"image": f"data:image/png;base64,{img_base64}"}

    except Exception as e:
        return {"error": f"Error generating visualization: {str(e)}"}
