import sys
import os
import io
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import PyPDF2

sys.path.append("./src")  
from chunking import chunk_by_tokens, chunk_by_sentence, chunk_by_paragraph, chunk_by_page
from similarity_metrics import SimilarityCalculator

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

def clean_text(text, use_stemming=True):
    """Make text searchable by cleaning it up"""
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    words = word_tokenize(text)
    stop_list = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_list]
    
    if use_stemming:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        
    return ' '.join(words)

def find_relevant_bit(text, query, max_len=10000):  # Set to a very high value
    """Return the full chunk content"""
    # First try to see if there are any matching paragraphs (for highlighting purposes)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    search_words = set(query.lower().split())
    
    # Check if there are matches
    has_matches = any(any(word in para.lower() for word in search_words) for para in paragraphs)
    
    # If there are matches or not, return the full text (only truncate if extremely long)
    if len(text) <= max_len:
        return text
    
    # Only truncate if extremely long
    return text[:max_len] + "... [truncated for display]"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            page_texts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                page_texts.append(page_text)
                text += page_text + "\n\n"
                
            return text, page_texts
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return "", []

class ChunkedTextSearcher:
    """Search through documents split into chunks"""
    
    def __init__(self, chunking_method="tokens", chunk_size=256, overlap=20):
        """
        Initialize the text searcher with specified chunking parameters
        
        Args:
            chunking_method: One of "tokens", "sentence", "paragraph", "page"
            chunk_size: Target size for token-based chunking
            overlap: Overlap size for token-based chunking
        """
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = None
        self.doc_names = []        
        self.docs = {}             
        self.pages = {}            
        self.chunks = {}           
        self.chunk_info = []       
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap
        
    def add_pdf(self, name, pdf_path):
        """Add a PDF document and split it into chunks"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                page_texts = []
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    
                    # Improve paragraph detection
                    if page_text:
                        # Add markers for headers to help paragraph detection
                        page_text = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z]*)+)(\n)', r'\1\n\n', page_text)
                    
                    page_texts.append(page_text)
                    text += page_text + "\n\n"
                    
                if not text:
                    print(f"Error: Could not extract text from {pdf_path}")
                    return False
                    
                self.docs[name] = text
                self.pages[name] = page_texts
                
                if name not in self.doc_names:
                    self.doc_names.append(name)
                
                # Split into chunks based on selected method
                if self.chunking_method == "tokens":
                    doc_chunks = chunk_by_tokens(text, token_size=self.chunk_size, overlap=self.chunk_overlap)
                elif self.chunking_method == "sentence":
                    doc_chunks = chunk_by_sentence(text, size=getattr(self, 'sentences_per_chunk', 1))
                elif self.chunking_method == "paragraph":
                    # For paragraph chunking, apply special pre-processing
                    modified_text = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-zA-Z]*)+)(\n)', r'\n\n\1\n\n', text)
                    doc_chunks = chunk_by_paragraph(modified_text)
                elif self.chunking_method == "page":
                    doc_chunks = page_texts
                else:
                    print(f"Unknown chunking method: {self.chunking_method}, using tokens")
                    doc_chunks = chunk_by_tokens(text, token_size=self.chunk_size, overlap=self.chunk_overlap)
                    
                self.chunks[name] = doc_chunks
                
                # Update chunk info list
                self._update_chunk_info()
                
                self.chunk_vectors = None
                
                print(f"Added PDF '{name}' with {len(doc_chunks)} chunks using {self.chunking_method} chunking.")
                return True
        except Exception as e:
            print(f"Error processing PDF {name}: {str(e)}")
            return False
    
    def add_text(self, name, content):
        """Add text content and split into chunks"""
        try:
            self.docs[name] = content
            
            self.pages[name] = [content]
            
            if name not in self.doc_names:
                self.doc_names.append(name)
            
            # Split into chunks based on selected method
            if self.chunking_method == "tokens":
                doc_chunks = chunk_by_tokens(content, token_size=self.chunk_size, overlap=self.chunk_overlap)
            elif self.chunking_method == "sentence":
                doc_chunks = chunk_by_sentence(content)
            elif self.chunking_method == "paragraph":
                doc_chunks = chunk_by_paragraph(content)
            elif self.chunking_method == "page":
                doc_chunks = [content]  
            else:
                print(f"Unknown chunking method: {self.chunking_method}, using tokens")
                doc_chunks = chunk_by_tokens(content, token_size=self.chunk_size, overlap=self.chunk_overlap)
                
            self.chunks[name] = doc_chunks
            
            self._update_chunk_info()
            
            self.chunk_vectors = None
            
            print(f"Added text '{name}' with {len(doc_chunks)} chunks using {self.chunking_method} chunking.")
            return True
        except Exception as e:
            print(f"Error adding text {name}: {str(e)}")
            return False
    
    def _update_chunk_info(self):
        """Update the chunk_info list with all chunks across all documents"""
        self.chunk_info = []
        for doc_name in self.doc_names:
            for i in range(len(self.chunks[doc_name])):
                self.chunk_info.append((doc_name, i))
    
    def build_index(self):
        """Create our search index from all document chunks"""
        if not self.chunk_info:
            print("No documents added yet.")
            return
            
        print(f"Building search index for {len(self.chunk_info)} chunks...")
            
        processed_chunks = []
        for doc_name, chunk_idx in self.chunk_info:
            chunk_text = self.chunks[doc_name][chunk_idx]
            processed_chunks.append(clean_text(chunk_text))
            
        self.chunk_vectors = self.vectorizer.fit_transform(processed_chunks)
        print("Search index built.")
    
    def search(self, query, num_results=3, similarity_metric="cosine"):
        """
        Find chunks matching a query
        
        Args:
            query: Search query string
            num_results: Number of results to return
            similarity_metric: One of "cosine", "euclidean", "jaccard"
            
        Returns:
            List of result dictionaries with document, chunk, and score information
        """
        if self.chunk_vectors is None:
            self.build_index()
            
        if self.chunk_vectors is None:
            print("No documents indexed.")
            return []
            
        clean_query = clean_text(query)
        
        query_vector = self.vectorizer.transform([clean_query])
        
        query_vector_array = query_vector.toarray()[0]
        chunk_vectors_array = self.chunk_vectors.toarray()
        
        # Calculate similarities using the SimilarityCalculator from the backend
        similarities = SimilarityCalculator.get_similarity_scores(
            query_vector_array, 
            chunk_vectors_array, 
            similarity_metric)
        
        results = []
        best_idx = np.argsort(similarities)[-num_results:][::-1]
        
        for idx in best_idx:
            doc_name, chunk_idx = self.chunk_info[idx]
            chunk_text = self.chunks[doc_name][chunk_idx]
            
            total_chunks = len(self.chunks[doc_name])
            position = (chunk_idx + 1) / total_chunks
            
            results.append({
                "document": doc_name,
                "chunk": chunk_idx + 1,
                "score": float(similarities[idx]),
                "position": position,
                "snippet": find_relevant_bit(chunk_text, query),
                "total_chunks": total_chunks,
                "chunk_text": chunk_text
            })
            
        return results
    
    def get_chunk_embeddings(self):
        """Get chunk vectors for visualization (to be implemented later)"""
        if self.chunk_vectors is None:
            self.build_index()
        
        return self.chunk_vectors.toarray(), self.chunk_info


def main():
    """Main function to demonstrate the chunked searcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunked PDF Search Tool")
    parser.add_argument("--pdf", help="Path to PDF file to analyze")
    parser.add_argument("--text", help="Path to text file to analyze")
    parser.add_argument("--chunking", choices=["tokens", "sentence", "paragraph", "page"], 
                       default="tokens", help="Chunking method to use")
    parser.add_argument("--chunk-size", type=int, default=256, 
                       help="Chunk size for token-based chunking")
    parser.add_argument("--overlap", type=int, default=20, 
                       help="Overlap size for token-based chunking")
    parser.add_argument("--query", help="Query to search for (optional)")
    parser.add_argument("--results", type=int, default=3, 
                       help="Number of results to return")
    parser.add_argument("--similarity", choices=["cosine", "euclidean", "jaccard"], 
                       default="cosine", help="Similarity metric to use")
    
    args = parser.parse_args()
    
    searcher = ChunkedTextSearcher(
        chunking_method=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    if args.pdf:
        if os.path.exists(args.pdf):
            doc_name = os.path.basename(args.pdf)
            success = searcher.add_pdf(doc_name, args.pdf)
            if not success:
                print(f"Failed to process PDF: {args.pdf}")
                sys.exit(1)
        else:
            print(f"PDF file not found: {args.pdf}")
            sys.exit(1)
            
    if args.text:
        if os.path.exists(args.text):
            doc_name = os.path.basename(args.text)
            with open(args.text, 'r', encoding='utf-8') as f:
                content = f.read()
            success = searcher.add_text(doc_name, content)
            if not success:
                print(f"Failed to process text file: {args.text}")
                sys.exit(1)
        else:
            print(f"Text file not found: {args.text}")
            sys.exit(1)
            
    if not args.pdf and not args.text:
        print("No input files provided. Use --pdf or --text to specify input.")
        sys.exit(1)
        
    # Build the index
    searcher.build_index()
    
    if args.query:
        print(f"\nSearching for: '{args.query}'")
        results = searcher.search(args.query, num_results=args.results, similarity_metric=args.similarity)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult #{i}:")
            print(f"Document: {result['document']}")
            print(f"Chunk: {result['chunk']}/{result['total_chunks']} (position: {result['position']:.2f})")
            print(f"Score: {result['score']:.4f}")
            print(f"Snippet: {result['snippet']}")
    else:
        print("\nNo search query provided. Use --query to search.")
        
        print("\nDocument summary:")
        for doc in searcher.doc_names:
            print(f"- {doc}: {len(searcher.chunks[doc])} chunks")


if __name__ == "__main__":
    main()