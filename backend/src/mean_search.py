import os
import re
import PyPDF2
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer  
from scipy.spatial.distance import jaccard
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)

class MeanSearcher:
    """A basic bag-of-words search implementation - first generation NLP approach"""
    
    def __init__(self):
        self.documents = {}  
        self.chunks = {}     
        self.chunk_info = []  
        self.vectorizer = CountVectorizer(  
            lowercase=True,
            binary=True,  # Use binary counts (1 or 0) for each word
            stop_words=None  
        )
        self.count_matrix = None  # Store bag-of-words counts
    
    def preprocess_text(self, text: str) -> str:
        """Very simple text preprocessing - just lowercase and remove punctuation"""
        text = text.lower()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def chunk_by_paragraph(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return paragraphs
    
    def chunk_by_sentence(self, text: str, sentences_per_chunk: int = 3) -> List[str]:
        """Split text into chunks of N sentences"""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)
        
        return chunks
    
    def chunk_by_fixed_size(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks of approximately fixed size"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def add_document(self, doc_name: str, content: str, chunking_method: str = 'paragraph', 
                    chunk_size: int = 200, overlap: int = 50, sentences_per_chunk: int = 3) -> int:
        """
        Add a document and split it into chunks using the specified method
        Returns the number of chunks created
        """
        self.documents[doc_name] = content
        
        processed_text = self.preprocess_text(content)
        
        if chunking_method == 'paragraph':
            chunks = self.chunk_by_paragraph(processed_text)
        elif chunking_method == 'sentence':
            chunks = self.chunk_by_sentence(processed_text, sentences_per_chunk)
        else:  
            chunks = self.chunk_by_fixed_size(processed_text, chunk_size, overlap)
        
        # Store the chunks
        self.chunks[doc_name] = chunks
        
        # Initialize chunk_info if needed
        if not hasattr(self, 'chunk_info'):
            self.chunk_info = []
        
        # Clear existing chunk_info for this document (if re-adding)
        self.chunk_info = [(d, i) for d, i in self.chunk_info if d != doc_name]
        
        # Update chunk_info
        for i in range(len(chunks)):
            self.chunk_info.append((doc_name, i))
        
        self.count_matrix = None
        
        return len(chunks)
    
    def add_pdf(self, doc_name: str, pdf_path: str, chunking_method: str = 'paragraph',
               chunk_size: int = 200, overlap: int = 50, sentences_per_chunk: int = 3) -> int:
        """
        Extract text from a PDF and add it as a document
        Returns the number of chunks created
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            return self.add_document(
                doc_name, text, chunking_method, chunk_size, overlap, sentences_per_chunk
            )
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return 0
    
    def build_index(self):
        """Build the bag-of-words matrix for all document chunks"""
        if not self.chunk_info:
            print("No documents added yet.")
            return
        
        all_chunks = []
        for doc_name, chunk_idx in self.chunk_info:
            all_chunks.append(self.chunks[doc_name][chunk_idx])
        
        # Create the bag-of-words matrix using CountVectorizer
        try:
            self.count_matrix = self.vectorizer.fit_transform(all_chunks)
            print(f"Built search index with {len(all_chunks)} chunks.")
        except Exception as e:
            print(f"Error building index: {str(e)}")
            # Fallback to even simpler configuration
            self.vectorizer = CountVectorizer(lowercase=True, binary=True)
            self.count_matrix = self.vectorizer.fit_transform(all_chunks)
            print(f"Built search index with fallback configuration.")
    
    def word_overlap_similarity(self, query_vector, chunk_vector):
        """
        Calculate simple word overlap similarity:
        (words in common) / (total unique words)
        This is a simple Jaccard similarity implementation for binary vectors
        """
        query_arr = query_vector.toarray().flatten()
        chunk_arr = chunk_vector.toarray().flatten()
        
        # For binary vectors, sum equals count of 1s
        intersection = np.sum(query_arr & chunk_arr)
        union = np.sum(query_arr | chunk_arr)
        
        # Avoid division by zero
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def search(self, query: str, num_results: int = 3, similarity_method: str = 'overlap') -> List[Dict[str, Any]]:
        """
        Search for chunks matching the query using basic bag-of-words matching
        
        Args:
            query: Search query string
            num_results: Number of results to return
            similarity_method: 'overlap' for word overlap (default), 'cosine' for cosine similarity
        
        Returns:
            List of results with document, chunk, and score information
        """
        if self.count_matrix is None:
            self.build_index()
            
        if self.count_matrix is None or len(self.chunk_info) == 0:
            return []
        
        # Preprocess the query (simple preprocessing)
        query = self.preprocess_text(query)
        
        # Transform query to bag-of-words vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity for each chunk
        similarities = []
        for i in range(self.count_matrix.shape[0]):
            chunk_vector = self.count_matrix[i:i+1]
            
            if similarity_method == 'overlap':
                similarity = self.word_overlap_similarity(query_vector, chunk_vector)
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(query_vector, chunk_vector)[0][0]
            
            similarities.append(similarity)
        
        top_indices = np.argsort(similarities)[-num_results:][::-1]
        
        results = []
        for idx in top_indices:
            doc_name, chunk_idx = self.chunk_info[idx]
            chunk_text = self.chunks[doc_name][chunk_idx]
            total_chunks = len(self.chunks[doc_name])
            
            snippet = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            
            results.append({
                "document": doc_name,
                "chunk": chunk_idx + 1,  
                "total_chunks": total_chunks,
                "score": float(similarities[idx]),
                "snippet": snippet,
                "position": (chunk_idx + 1) / total_chunks
            })
        
        return results