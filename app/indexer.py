import numpy as np
import faiss


class CodeIndexer:
    def __init__(self, dim=768):
        self.dim = dim
        self.index = None
        self.metadata_store = []

    def build_index(self, chunks):
        """
        Builds the FAISS index from a list of chunk dicts (each with an 'embedding' field).
        Strips embeddings out of the chunk dicts before storing them as metadata.
        """
        embeddings = []
        metadata_store = []

        for chunk in chunks:
            embeddings.append(chunk.pop('embedding'))
            metadata_store.append(chunk)

        embedding_store = np.array(embeddings).astype('float32')

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embedding_store)
        self.metadata_store = metadata_store

        return self.index, self.metadata_store

    def search(self, query_embedding, k=5):
        """
        Takes a pre-embedded query vector (from CodeEmbedder.embed_query),
        returns top-k chunk metadata dicts with similarity scores.
        """
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for rank, i in enumerate(indices[0]):
            results.append({**self.metadata_store[i], 'score': float(distances[0][rank])})

        return results