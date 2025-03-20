from fastapi import FastAPI, File, UploadFile, Form, Query
import io
import PyPDF2
from fastapi.middleware.cors import CORSMiddleware
from chunking import chunk_by_sentence, chunk_by_paragraph, chunk_by_page, chunk_by_tokens
import os
import json
import sys
import numpy as np
import tempfile
from embedding import EmbeddingGenerator
from similarity_metrics import SimilarityCalculator
from utils import format_context_for_llm, generate_llm_response
from visualization import PCA_visualization, tSNE_visualization, UMAP_visualization
from naive import ChunkedTextSearcher
from mean_search import MeanSearcher

app = FastAPI(root_path='/api')

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

origins = [
    "http://localhost:5173",
    "http://vcm-45508.vm.duke.edu",
    "http://localhost:5174",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)

current_document = {
    "text": "",
    "pages": [],
    "chunks": [],
    "chunking_strategy": "",
    "embedding_model": "",
    "embeddings": [],
    "similarity_metric": "",
    "embeddings_file": ""
}

mean_searcher = MeanSearcher()
naive_searcher = ChunkedTextSearcher(chunking_method="tokens", chunk_size=256, overlap=20)
query_embedding = []
similarity_scores = [] 
llm_response = ""

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
        config_id = f"{file.filename}_{chunking_strategy}_{token_size}_{sentence_size}_{paragraph_size}_{page_size}_{embedding_model}"
        
        # Load existing configurations if any
        config_file = "data/current_document.json"
        configs = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                configs = json.load(f)
        
        # Check if we have embeddings for this configuration
        if config_id in configs:
            # Load existing embeddings
            embeddings_file = configs[config_id]["embeddings_file"]
            with open(embeddings_file, "r") as f:
                chunk_embeddings = json.load(f)
                chunks = [item["chunk"] for item in chunk_embeddings]
                embeddings = [item["embeddings"] for item in chunk_embeddings]
                
                # Load the PDF to get pages
                pdf_content = await file.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                pages = [page.extract_text() for page in pdf_reader.pages]
                full_text = "\n\n".join(pages)
                
                current_document.update({
                    "text": full_text,
                    "pages": pages,
                    "chunks": chunks,
                    "chunking_strategy": chunking_strategy,
                    "embedding_model": embedding_model,
                    "embeddings": embeddings,
                    "similarity_metric": similarity_metric,
                    "embeddings_file": embeddings_file  # Store the embeddings file path
                })
                
                return {
                    "message": f"Loaded existing embeddings for configuration. Found {len(chunks)} chunks.",
                    "chunk_count": len(chunks)
                }
            
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        full_text = ""
        pages = []
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            pages.append(page_text)
            full_text += page_text + "\n\n"
        
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

        current_document.update({
            "text": full_text,
            "pages": pages,
            "chunks": chunks,
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model,
            "embeddings": embeddings,
            "similarity_metric": similarity_metric,
        })
        
        # Save embeddings to a unique file
        embeddings_file = f"data/embeddings_{config_id}.json"
        chunk_embeddings = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_embeddings.append({
                "chunk": chunk,
                "chunk_no": i+1,
                "embeddings": embedding
            })
        with open(embeddings_file, "w") as f:
            json.dump(chunk_embeddings, f)

        # Update the config to include the embeddings file path
        configs[config_id] = {
            "embeddings_file": embeddings_file,
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model,
            "similarity_metric": similarity_metric,
        }
        
        # Save the updated configurations
        with open(config_file, "w") as f:
            json.dump(configs, f)

        return {
            "message": f"PDF processed with {chunking_strategy} chunking strategy and {embedding_model} model. Created {len(chunks)} chunks.",
            "chunk_count": len(chunks)
        }
    
    except Exception as e:
        return {"message": str(e)}

@app.post("/upload-mean")
async def upload_mean(
    file: UploadFile = File(...),
    chunking_method: str = Form("paragraph"),
    chunk_size: int = Form(200),
    overlap: int = Form(50),
    sentences_per_chunk: int = Form(3)
):
    """
    Upload a document for the mean (lexical) search approach
    """
    global mean_searcher
    
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            # Save PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            num_chunks = mean_searcher.add_pdf(
                file.filename, temp_path, 
                chunking_method, chunk_size, overlap, sentences_per_chunk
            )
            
            os.unlink(temp_path)
        else:
            text_content = content.decode('utf-8')
            num_chunks = mean_searcher.add_document(
                file.filename, text_content,
                chunking_method, chunk_size, overlap, sentences_per_chunk
            )
        
        mean_searcher.build_index()
        
        return {
            "message": f"Document processed with {chunking_method} chunking. Created {num_chunks} chunks."
        }
    
    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.get("/mean")
def query_mean_model(query: str):
    """
    Query endpoint for the mean model
    """
    # Pass query to some function
    answer = f"Response to the mean query : {query}"
    return {"answer": answer}

@app.get("/search-mean")
async def search_mean(
    query: str,
    num_results: int = 3,
    similarity_method: str = "overlap"
):
    """
    Search using the mean (first-generation bag-of-words) approach
    """
    global mean_searcher
    
    try:
        results = mean_searcher.search(query, num_results, similarity_method)
        
        if results:
            top_result = results[0]
            answer = f"Found {len(results)} relevant chunks."
        else:
            answer = "No relevant results found for your query."
        
        return {
            "answer": answer,
            "results": results
        }
    
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "results": []
        }

@app.get("/traditional")
def query_traditional_model(query: str):
    """
    Query endpoint for the traditional model
    """
    answer = f"Response to the traditional query : {query}"
    return {"answer": answer}

@app.post("/upload-naive")
async def upload_naive(
    file: UploadFile = File(...),
    chunking_method: str = Form("tokens"),
    chunk_size: int = Form(256),
    overlap: int = Form(20),
    similarity_metric: str = Form("cosine"),
):
    """
    Upload a document for the naive chunked text search approach
    """
    global naive_searcher
    
    try:
        naive_searcher.chunking_method = chunking_method
        naive_searcher.chunk_size = chunk_size
        naive_searcher.chunk_overlap = overlap
        
        # Read the file
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            success = naive_searcher.add_pdf(file.filename, temp_path)
            
            os.unlink(temp_path)
            
            if not success:
                return {"message": f"Error processing PDF: {file.filename}"}
        else:
            text_content = content.decode('utf-8')
            success = naive_searcher.add_text(file.filename, text_content)
            
            if not success:
                return {"message": f"Error processing text file: {file.filename}"}
        
        naive_searcher.build_index()
        
        chunk_count = sum(len(chunks) for chunks in naive_searcher.chunks.values())
        return {
            "message": f"Document processed with {chunking_method} chunking. Created {chunk_count} chunks."
        }
    
    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.get("/search-naive")
async def search_naive(
    query: str,
    num_results: int = 3,
    similarity_metric: str = "cosine"
):
    """
    Search using the naive chunked text search approach
    """
    global naive_searcher
    
    try:
        # Perform the search
        results = naive_searcher.search(query, num_results=num_results, similarity_metric=similarity_metric)
        
        if results:
            top_result = results[0]
            answer = f"Found {len(results)} relevant chunks. Most relevant (score: {top_result['score']:.2f}) is from document '{top_result['document']}'."
        else:
            answer = "No relevant results found for your query."
        
        return {
            "answer": answer,
            "results": results
        }
    
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "results": []
        }

@app.get("/deep-learning")
def query_deep_learning_model(query: str, num_chunks: int = 5):
    """
    Query endpoint for the deep learning model
    """

    global query_embedding, similarity_scores

    if not current_document["chunks"]:
        return {"answer": "Please upload a document first."}
    
    try:
        # Get embeddings for the query using the same model as the chunks
        query_embedding = EmbeddingGenerator.get_embeddings([query], current_document["embedding_model"])[0]
        
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
        
        top_chunk_texts = [(chunk, chunk_number) for chunk_number, (chunk, _) in top_chunks]
        context = format_context_for_llm(top_chunk_texts)
        llm_response = generate_llm_response(query, context)
        
        answer = llm_response
        
        chunks_data = [
            {
                "chunk_number": chunk_idx + 1, 
                "text": chunk,
                "relevance_score": float(score)  
            }
            for chunk_idx, (chunk, score) in top_chunks
        ]
            
        return {
            "answer": answer,
            "chunks": chunks_data
        }
    
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
        response_emb = EmbeddingGenerator.get_embeddings([llm_response], current_document["embedding_model"])[0]

        top_chunk_indices = np.argsort(similarity_scores)[-k:]

        if method.lower() == "pca":
            img_base64 = PCA_visualization(chunks_embs, query_emb, response_emb, top_chunk_indices)
        elif method.lower() == "tsne":
            img_base64 = tSNE_visualization(chunks_embs, query_emb, response_emb, top_chunk_indices)
        elif method.lower() == "umap":
            img_base64 = UMAP_visualization(chunks_embs, query_emb, response_emb, top_chunk_indices)
        else:
            return {"error": "Invalid visualization method"}

        return {"image": f"data:image/png;base64,{img_base64}"}

    except Exception as e:
        return {"error": f"Error generating visualization: {str(e)}"}