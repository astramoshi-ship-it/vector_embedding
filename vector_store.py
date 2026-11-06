
import os
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Create folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DOCUMENTS_DIR = os.path.join(DATASET_DIR, "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# 2. Load all text documents
file_names = sorted(
    f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".txt")
)

documents = []
ids = []
metadatas = []

for i, filename in enumerate(file_names):
    file_path = os.path.join(DOCUMENTS_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as file:
        documents.append(file.read())

    ids.append(f"doc{i+1}")

    metadatas.append({
        "title": filename.replace(".txt", ""),
        "category": "E-Commerce",
        "source": filename
    })

print("Total documents loaded:", len(documents))

# 3. Load Sentence Transformer model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Generate embeddings
embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    show_progress_bar=True
)

# Save embeddings
np.save(
    os.path.join(DATASET_DIR, "embeddings.npy"),
    embeddings
)

print("Embeddings generated successfully!")
print("Embedding shape:", embeddings.shape)

# 5. Create ChromaDB database
client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="ecommerce_documents",
    metadata={"hnsw:space": "cosine"}
)

# 6. Store documents and embeddings
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Documents stored in ChromaDB!")

# 7. Create user queries
queries = [
    "How can I shop online?",
    "Where can I find electronic products?",
    "What is product information?",
    "How can I get discounts?",
    "Why are customer reviews important?",
    "What are online payment methods?",
    "What happens after placing an order?",
    "How does product delivery work?",
    "How can I return a damaged product?",
    "How can I contact customer support?"
]

query_df = pd.DataFrame({
    "query_id": [f"Q{i+1}" for i in range(len(queries))],
    "query": queries
})

query_df.to_csv(
    os.path.join(DATASET_DIR, "queries.csv"),
    index=False
)

# 8. Perform similarity search
all_results = []

for i, query in enumerate(queries):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, len(documents))
    )

    # Keyword-based search
    query_words = set(query.lower().split())
    keyword_scores = []

    for doc_index, document in enumerate(documents):
        doc_words = set(document.lower().split())
        score = len(query_words.intersection(doc_words))
        keyword_scores.append((doc_index, score))

    keyword_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    keyword_ranks = {
        ids[index]: rank
        for rank, (index, score) in enumerate(
            keyword_scores[:5], start=1
        )
    }

    # Save top 5 similarity results
    for rank in range(len(results["ids"][0])):

        doc_id = results["ids"][0][rank]
        doc_text = results["documents"][0][rank]
        distance = results["distances"][0][rank]
        metadata = results["metadatas"][0][rank]

        all_results.append({
            "query_id": f"Q{i+1}",
            "query": query,
            "document_id": doc_id,
            "document_title": metadata["title"],
            "similarity_rank": rank + 1,
            "cosine_distance": round(float(distance), 4),
            "keyword_rank": keyword_ranks.get(
                doc_id, "Not in Top 5"
            ),
            "retrieved_document": doc_text
        })

# 9. Save retrieved results
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(DATASET_DIR, "retrieved_results.csv"),
    index=False
)

# 10. Display output
print("\nSimilarity Search Completed!")
print("Total queries:", len(queries))
print("Total retrieved results:", len(results_df))

print("\nSample Results:")
print(
    results_df[
        ["query_id", "document_title",
         "similarity_rank", "keyword_rank"]
    ].head(10).to_string(index=False)
)

print("\nAll output files saved successfully!")
print("Documents stored in ChromaDB!")
# Verify stored documents
print("\n--- ChromaDB Verification ---")

print("Total records stored:", collection.count())

stored_data = collection.get(
    limit=10,
    include=["documents", "metadatas"]
)

for i in range(len(stored_data["ids"])):
    print("\nDocument ID:", stored_data["ids"][i])
    print("Metadata:", stored_data["metadatas"][i])
    print("Content:", stored_data["documents"][i])

print("\nChromaDB verification completed!")