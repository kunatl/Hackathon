import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("high_quality_it_tickets.csv")
df["text"] = df["title"] + " " + df["description"]

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = embed_model.encode(df["text"].tolist())
embeddings = np.array(embeddings).astype("float32")

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# 🔍 Retrieve similar tickets
def search_similar_tickets(query, k=3):
    query_vec = embed_model.encode([query])
    query_vec = np.array(query_vec).astype("float32")
    
    distances, indices = index.search(query_vec, k)
    
    results = []
    for i in indices[0]:
        results.append({
            "title": df.iloc[i]["title"],
            "description": df.iloc[i]["description"],
            "resolution": df.iloc[i]["resolution"],
            "category": df.iloc[i]["category"]
        })
    
    return results


from openai import OpenAI

import os

client = OpenAI(
    api_key="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",   # paste your key
    base_url="https://openrouter.ai/api/v1"
)

def generate_resolution(query, results):
    context = "\n\n".join([
        f"Ticket: {r['description']}\nResolution: {r['resolution']}"
        for r in results
    ])
    
    prompt = f"""
    You are an IT support assistant.

    Ticket:
    {query}

    Similar cases:
    {context}

    Provide step-by-step resolution.
    """
    
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


# 🔁 Main RAG function
def suggest_resolution(query):
    results = search_similar_tickets(query, k=3)
    
    try:
        resolution = generate_resolution(query, results)
    except Exception as e:
        print("⚠️ LLM failed → using fallback")
        
        resolutions = [r["resolution"] for r in results]
        resolution = max(set(resolutions), key=resolutions.count)
    
    return resolution, results