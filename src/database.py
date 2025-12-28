import chromadb
from chromadb.config import Settings
from src.config import DB_PATH

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)

        self.paper_collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )

        self.image_collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )

    def add_paper(self, doc_id, vector, metadata):
        self.paper_collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[metadata]
        )

    def add_image(self, img_id, vector, metadata):
        self.image_collection.upsert(
            ids=[img_id],
            embeddings=[vector],
            metadatas=[metadata]
        )

    def search_papers(self, query_vector, top_k=5):
        return self.paper_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

    def search_images(self, query_vector, top_k=5):
        return self.image_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )