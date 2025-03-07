import numpy as np
from typing import List, Tuple, Dict, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, jaccard

def calculate_cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: List[np.ndarray]) -> List[float]:
    """
    Calculate cosine similarity between query embedding and chunk embeddings.
    """
    # Reshape query embedding to (1, embedding_dim)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Convert chunk embeddings to numpy array if not already
    chunk_embeddings = np.array(chunk_embeddings)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    return similarities.tolist()

def calculate_euclidean_similarity(query_embedding: np.ndarray, chunk_embeddings: List[np.ndarray]) -> List[float]:
    """
    Calculate similarity based on Euclidean distance between query embedding and chunk embeddings.
    """
    query_embedding = np.array(query_embedding)
    similarities = []
    
    for chunk_emb in chunk_embeddings:
        # Calculate Euclidean distance
        dist = euclidean(query_embedding, chunk_emb)
        # Convert distance to similarity (1 / (1 + distance))
        # This maps the range [0, inf) to (0, 1]
        similarity = 1 / (1 + dist)
        similarities.append(similarity)
    
    return similarities

def calculate_jaccard_similarity(query_embedding: np.ndarray, chunk_embeddings: List[np.ndarray]) -> List[float]:
    """
    Calculate Jaccard similarity between query embedding and chunk embeddings.
    For continuous embeddings, we convert them to binary by thresholding at median value.
    """
    query_embedding = np.array(query_embedding)
    # Convert to binary using median as threshold
    query_median = np.median(query_embedding)
    query_binary = query_embedding > query_median
    
    similarities = []
    for chunk_emb in chunk_embeddings:
        chunk_emb = np.array(chunk_emb)
        chunk_median = np.median(chunk_emb)
        chunk_binary = chunk_emb > chunk_median
        
        # Calculate Jaccard similarity (1 - Jaccard distance)
        similarity = 1 - jaccard(query_binary, chunk_binary)
        similarities.append(similarity)
    
    return similarities

def get_similarity_scores(query_embedding: np.ndarray, 
                          chunk_embeddings: List[np.ndarray], 
                          similarity_metric: str = "cosine") -> List[float]:
    """
    Calculate similarity scores based on specified metric.
    """
    if similarity_metric == "cosine":
        return calculate_cosine_similarity(query_embedding, chunk_embeddings)
    elif similarity_metric == "euclidean":
        return calculate_euclidean_similarity(query_embedding, chunk_embeddings)
    elif similarity_metric == "jaccard":
        return calculate_jaccard_similarity(query_embedding, chunk_embeddings)
    else:
        # Default to cosine similarity
        return calculate_cosine_similarity(query_embedding, chunk_embeddings)

def get_top_k_chunks(chunks: List[str], 
                    similarity_scores: List[float], 
                    k: int = 5) -> List[Tuple[str, float]]:
    """
    Retrieve top k chunks based on similarity scores.
    """
    # Create list of (chunk, score) tuples
    chunk_scores = list(zip(chunks, similarity_scores))
    # Sort by score in descending order
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    # Return top k
    return sorted_chunks[:k]