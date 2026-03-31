import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("high_quality_it_tickets.csv")

# Combine text
df["text"] = df["title"] + " " + df["description"]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert tickets to embeddings
embeddings = model.encode(df["text"].tolist())

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

print("✅ FAISS index created with", index.ntotal, "tickets")


def search_similar_tickets(query, k=3):
    query_vec = model.encode([query])
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

def suggest_resolution(query):
    results = search_similar_tickets(query, k=3)
    
    resolutions = [r["resolution"] for r in results]
    
    # Simple strategy: most frequent resolution
    final_resolution = max(set(resolutions), key=resolutions.count)
    
    return final_resolution, results

if __name__ == "__main__":
    query = input("Enter new ticket:\n")
    
    results = search_similar_tickets(query)
    
    print("\n🔍 Top Similar Tickets:\n")
    
    for i, r in enumerate(results, 1):
        print(f"\n--- Match {i} ---")
        print("Title:", r["title"])
        print("Category:", r["category"])
        print("Resolution:", r["resolution"])

    final_resolution, results = suggest_resolution(query)
    print("\n💡 Suggested Resolution:")
    print(final_resolution)