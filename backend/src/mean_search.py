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
    nltk.download('punkt_tab', quiet=True)

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
        """
        Enhanced text preprocessing for PDF content
        """
        # First, normalize line endings
        text = re.sub(r'\r\n', '\n', text)
        
        # Normalize spacing
        text = re.sub(r' +', ' ', text)
        
        # Add double newlines before headings (likely section titles)
        text = re.sub(r'([a-z])\n([A-Z][a-zA-Z ]+)(\n)', r'\1\n\n\2\3', text)
        
        return text
    
    def chunk_by_paragraph(self, text: str) -> List[str]:
        """
        Vastly improved paragraph detection specifically for PDF content
        """
        print(f"DEBUG: chunk_by_paragraph input length: {len(text)} characters")
        
        # First normalize text
        text = self.preprocess_text(text)
        
        # Method 1: Standard paragraph detection
        paragraphs1 = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        print(f"DEBUG: Standard paragraph detection found {len(paragraphs1)} paragraphs")
        
        # Method 2: Enhanced detection looking for section headers
        enhanced_text = re.sub(r'([A-Z][a-zA-Z ]*(?:[a-z]|[A-Z][a-z]+)[a-zA-Z ]*)\n', r'\n\n\1\n\n', text)
        paragraphs2 = [p.strip() for p in re.split(r'\n\s*\n', enhanced_text) if p.strip()]
        print(f"DEBUG: Enhanced paragraph detection found {len(paragraphs2)} paragraphs")
        
        # Use whichever method found more paragraphs
        paragraphs = paragraphs1 if len(paragraphs1) >= len(paragraphs2) else paragraphs2
        
        # If still only one paragraph, force split into smaller chunks
        if len(paragraphs) <= 1 and len(text) > 500:
            # If the text is very long but has no proper paragraph breaks, 
            # split by sentence and then group sentences
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            from nltk.tokenize import sent_tokenize
            
            sentences = sent_tokenize(text)
            print(f"DEBUG: Document has {len(sentences)} sentences")
            
            # Group sentences into pseudo-paragraphs (5 sentences per paragraph)
            paragraphs = []
            for i in range(0, len(sentences), 5):
                paragraph = " ".join(sentences[i:i + 5])
                paragraphs.append(paragraph)
            
            print(f"DEBUG: Created {len(paragraphs)} forced paragraphs from sentences")
        
        # As a final failsafe, if we still have no good paragraphs, split by fixed length
        if len(paragraphs) <= 1 and len(text) > 500:
            print("DEBUG: Using fixed length chunking as fallback")
            words = text.split()
            paragraphs = []
            
            for i in range(0, len(words), 100):  # ~100 words per paragraph as fallback
                paragraph = " ".join(words[i:i + 100])
                paragraphs.append(paragraph)
        
        # Print sample of the first paragraph
        if paragraphs:
            print(f"DEBUG: First paragraph sample: {paragraphs[0][:100]}...")
        
        return paragraphs
    
    def chunk_by_sentence(self, text: str, sentences_per_chunk: int = 3) -> List[str]:
        """
        Completely rewritten sentence chunking for better reliability
        """
        print(f"DEBUG: chunk_by_sentence with {sentences_per_chunk} sentences per chunk")
        
        # Ensure NLTK punkt is downloaded
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        
        # Process text to improve sentence detection
        text = self.preprocess_text(text)
        
        # First try standard sentence tokenization
        sentences = sent_tokenize(text)
        print(f"DEBUG: Detected {len(sentences)} sentences with NLTK tokenizer")
        
        # If NLTK doesn't find many sentences, try a simple regex approach
        if len(sentences) <= 5 and len(text) > 500:
            print("DEBUG: Few sentences detected, trying regex approach")
            # Simple regex to split on sentence endings
            regex_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            if len(regex_sentences) > len(sentences):
                sentences = regex_sentences
                print(f"DEBUG: Regex found {len(sentences)} sentences")
        
        # Remove very short sentences (likely artifacts)
        sentences = [s for s in sentences if len(s) > 10]
        print(f"DEBUG: After filtering, {len(sentences)} valid sentences remain")
        
        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)
        
        print(f"DEBUG: Created {len(chunks)} chunks from sentences")
        
        # As a failsafe, if we have no chunks, return the original text as one chunk
        if not chunks and text.strip():
            print("DEBUG: No chunks created, returning original text as one chunk")
            return [text]
        
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
                 chunk_size: int = 200, overlap: int = 50, sentences_per_chunk: int = 1) -> int:
        """Add a document and split it into chunks using the specified method"""
        self.documents[doc_name] = content
        
        processed_text = self.preprocess_text(content)
        
        # Add debug print
        print(f"Using chunking method: {chunking_method}")
        
        if chunking_method == 'paragraph':
            chunks = self.chunk_by_paragraph(processed_text)
        elif chunking_method == 'sentence':
            # Use custom sentence chunker to ensure proper handling
            chunks = self.chunk_by_sentence(processed_text, sentences_per_chunk)
        else:  # fixed_size
            chunks = self.chunk_by_fixed_size(processed_text, chunk_size, overlap)
        
        # More debug info
        print(f"Created {len(chunks)} chunks for document {doc_name}")
        
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
        Enhanced PDF text extraction
        """
        try:
            print(f"DEBUG: add_pdf with chunking_method={chunking_method}, sentences_per_chunk={sentences_per_chunk}")
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                print(f"DEBUG: PDF has {len(reader.pages)} pages")
                
                text = ""
                page_texts = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    page_texts.append(page_text)
                    
                    if page_text:
                        # Add explicit paragraph markers for better detection
                        page_text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', page_text)
                        text += page_text + "\n\n"  # Add explicit paragraph breaks between pages
                    
                    print(f"DEBUG: Page {i+1} has {len(page_text) if page_text else 0} characters")
                
                print(f"DEBUG: Total text extracted: {len(text)} characters")
                
                # Create a version of text with enhanced formatting for debugging
                enhanced_text = re.sub(r'\n', '<NEWLINE>\n', text[:1000])
                print(f"DEBUG: Text sample with newlines marked: {enhanced_text[:500]}...")
                
                # Call add_document with the extracted text
                result = self.add_document(
                    doc_name, text, chunking_method, chunk_size, overlap, sentences_per_chunk
                )
                
                return result
                
        except Exception as e:
            print(f"ERROR processing PDF: {str(e)}")
            import traceback
            print(traceback.format_exc())
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