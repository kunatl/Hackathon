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


# 🤖 FULL RAG (Gemini Generation)
from google import genai

# Initialize client
client = genai.Client(api_key="AIzaSyCVKeu0cq0WJKywu7K4OhOsjkOqGF9Uusw")


def generate_resolution_with_gemini(query, results):
    if not results:
        return "No similar tickets found. Escalate to support."
    
    context = "\n\n".join([
        f"Ticket: {r['description']}\nResolution: {r['resolution']}"
        for r in results
    ])
    
    prompt = f"""
    You are an expert IT support assistant.

    New Ticket:
    {query}

    Similar Past Tickets:
    {context}

    Provide a clear, step-by-step resolution.
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # ✅ correct working model
        contents=prompt
    )
    
    return response.text


# 🔁 Main RAG function
def suggest_resolution(query):
    results = search_similar_tickets(query, k=3)
    resolution = generate_resolution_with_gemini(query, results)
    return resolution, results