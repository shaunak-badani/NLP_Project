# RAG Evaluation Application

## Overview
This application provides an interactive environment for testing various RAG configurations and visualizing their performance, making RAG systems more explainable and transparent.

This application aims to:

1. Provide a configurable testing environment for RAG components
2. Visualize embedding spaces and relationships between queries and documents
3. Enable transparent evaluation of different RAG configurations
4. Facilitate better understanding of how retrieval mechanisms work

## Features

### Configurable Components for Comprehensive Testing
- **Multiple Chunking Strategies**:
  - Sentence-based chunking (with configurable sentences per chunk)
  - Paragraph-based chunking (with configurable paragraphs per chunk)
  - Page-based chunking (with configurable pages per chunk)
  - Token-based chunking (with configurable token count)
  
- **Diverse Embedding Models**:
  - Sentence-Transformer (all-MiniLM-L6-v2)
  - BERT (bert-base-uncased)
  - RoBERTa (roberta-base)
  - DistilBERT (distilbert-base-uncased)
  - GPT-2 (gpt2)
  - Fine-tuned Financial model (bge-base)

- **Multiple Similarity Metrics**:
  - Cosine similarity
  - Euclidean similarity
  - Jaccard similarity

### Rich Validation Interface
- **Interactive Chunk Exploration**:
  - Expandable view of retrieved chunks
  - Relevance scores for each chunk
  - Markdown rendering of chunk content
  
- **Multi-dimensional Visualization**:
  - Principal Component Analysis (PCA)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Uniform Manifold Approximation and Projection (UMAP)
  - Visual validation of embedding space relationships

- **Transparent Retrieval Process**:
  - Clear display of ranked chunks and their scores
  - LLM response based on retrieved context
  - Complete transparency of the retrieval-to-response pipeline

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Required Python packages (see `requirements.txt`)
- Required NPM packages (see `package.json`)

### Installation

1. Clone the repository and create a virtual environment.

2. Install backend dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

### Running the Application

1. Start the backend server:
```bash
cd backend
fastapi dev src/main.py
```

2. Start the frontend development server:
```bash
npm run dev
```

## Usage Guide

### Document Upload and Configuration
1. Upload a PDF document using the file selector
2. Choose a chunking strategy (sentence, paragraph, page, or tokens)
3. Configure the chunking parameters based on your selection
4. Select an embedding model
5. Choose a similarity metric
6. Set the number of chunks to retrieve
7. Click "Upload and Process" to prepare the document

### Querying the Document
1. Enter your query in the text area
2. Click "Send Query" to process the query
3. Review the LLM response generated from the retrieved context
4. Explore the retrieved chunks, ranked by relevance
5. Click "Create visualization" to generate embedding space visualizations

### Analyzing Results
1. Examine the retrieved chunks and their relevance scores
2. Review the visualizations to understand the spatial relationships between:
   - Document chunks (all chunks)
   - Query embedding
   - Top-k relevant chunks
   - LLM response embedding
3. Compare different configurations to determine optimal settings

## Future Roadmap

### Comparative Analysis Framework
- Simultaneous evaluation of multiple configuration combinations
- Side-by-side comparison of different chunking strategies, embedding models, and similarity metrics
- Performance metrics across different settings

### LLM-as-a-Judge
- Automated assessment of retrieval quality
- Using LLMs as objective judges for:
  - Context relevance
  - Information completeness
  - Response accuracy
  - And more evaluation criteria


# Ethics Statement

### Transparency and Explainability

Our application is fundamentally designed to enhance transparency in RAG systems. We recognize that:

- RAG systems can be opaque, making it difficult to understand why certain information was retrieved and how it influenced the final output
- Users deserve to know how their queries are being processed and what factors influence the information they receive

We address these concerns by:
- Providing detailed visualizations of embedding spaces
- Displaying relevance scores for retrieved chunks
- Enabling the inspection of all retrieved context

### Bias Mitigation and Fairness

Our application supports bias mitigation efforts by:
- Allowing comparison between different embedding models to identify potential biases
- Making retrieval decisions explicit and reviewable

## Limitations and Responsible Use

We acknowledge the following limitations of our application:

1. **Visualization limitations**: Dimensionality reduction techniques necessarily simplify complex relationships and may not capture all relevant aspects of the embedding space
2. **Evaluation scope**: Our application focuses on retrieval performance and does not fully address all aspects of RAG system quality
3. **Model constraints**: The performance of the system depends on the quality of the embedding models and LLMs being used
4. **Domain specificity**: Performance in specialized domains may require additional consideration and domain-specific evaluation