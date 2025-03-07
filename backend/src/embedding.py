from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, GPT2Model, GPT2Tokenizer, T5Tokenizer, T5Model
import torch
import numpy as np
import os
import json

# Load models globally to avoid reloading per request
sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2")

def get_embeddings(texts, model_name):

    embeddings = []
    
    if model_name == "sentence-transformer":
        embeddings = sentence_transformer.encode(texts).tolist()
    elif model_name == "bert":
        inputs = bert_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()
    elif model_name == "roberta":
        inputs = roberta_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = roberta_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()
    elif model_name == "distilbert":
        inputs = distilbert_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = distilbert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()
    elif model_name == "gpt2":
        inputs = gpt2_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = gpt2_model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1, :].detach().numpy().tolist()
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")
    
    return embeddings
