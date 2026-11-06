
import os
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="ecommerce_documents"
)

print("Database verification started")

print("Total documents:", collection.count())

data = collection.get(
    include=["documents", "metadatas"]
)

for i in range(len(data["ids"])):
    print("\nID:", data["ids"][i])
    print("Document:", data["documents"][i])
    print("Metadata:", data["metadatas"][i])

