from typing import List, Dict, Any
import os
import json
import requests
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

def format_context_for_llm(chunks: List[str]) -> str:
    """
    Format chunks as context for the LLM.
    """
    context = ""
    for chunk, chunk_no in chunks:
        context += f"Chunk {chunk_no+1}:\n{chunk}\n\n"

    return f"CONTEXT:\n{context}\n\nBased on the above context, "

def generate_llm_response(query: str, context: str) -> str:
    """
    Generate a response from Google Gemini API based on the query and context.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            return "Error: Gemini API key not found. Please set GEMINI_API_KEY in your environment variables."

        client = genai.Client(api_key=api_key)
        prompt = f"""{context}

Based purely on the above context:

1. Answer the following question: {query}

2. After your answer, provide a detailed relevance analysis:
   - List each chunk number that was used to answer the question
   - Assign a relevance score (0-100%) to each chunk. Be strict with the relevance score and give appropriate scores.
   - Ensure the sum of all relevance scores equals 100%
   - Note: Chunks can have 0% (not used) or 100% (fully used) relevance
"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
)
        
        # Return the response text
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"