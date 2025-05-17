import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# ==== CONFIG ====
ARXIV_FILE = 'datasets\arxiv_physics_subset.json'  
INDEX_FILE = 'physics_faiss.index'         
EMBEDDINGS_FILE = 'physics_embeddings.npy' 
META_FILE = 'physics_metadata.json'        
MODEL_NAME = 'all-MiniLM-L6-v2'            

# ==== LOAD DATA ====
with open(ARXIV_FILE, 'r') as f:
    papers = json.load(f)

documents = [f"{p['title'].strip()}: {p['abstract'].strip()}" for p in papers]

# ==== EMBEDDING ====
print("Loading model and embedding...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)

# ==== FAISS INDEX ====
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 distance
index.add(embeddings)

# ==== SAVE INDEX AND METADATA ====
print(f"Saving index to: {INDEX_FILE}")
faiss.write_index(index, INDEX_FILE)
np.save(EMBEDDINGS_FILE, embeddings)

with open(META_FILE, 'w') as f:
    json.dump(papers, f, indent=2)
