import re
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import tiktoken

def chunk_by_sentence(text):
    """Split text into chunks by sentence"""
    sentences = sent_tokenize(text)
    return sentences

def chunk_by_paragraph(text):
    """Split text into chunks by paragraph"""
    # Split by double newlines or more to separate paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    # Filter out empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def chunk_by_page(text, page_texts):
    """Return chunks by page (using pre-separated page texts)"""
    return page_texts

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_by_tokens(text, token_size=256, overlap=20, encoding_name="cl100k_base"):
    """Split text into chunks of approximately token_size tokens with optional overlap"""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        # Get chunk of tokens
        chunk_end = min(i + token_size, len(tokens))
        chunk_tokens = tokens[i:chunk_end]
        
        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move to next chunk, considering overlap
        i += token_size - overlap
    
    return chunks