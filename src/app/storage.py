import os
import pickle
import numpy as np
import faiss

META_FILE_NAME = "meta.pkl"


class VectorStore:
    def __init__(self, dim=384):
        # Path where FAISS index and metadata are stored
        self.dim = dim
        self.index_path = os.getenv("VECTOR_DB_PATH", "/data/faiss_index")
        self.meta_path = os.path.join(os.path.dirname(self.index_path), META_FILE_NAME)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Load existing FAISS index and metadata if available
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.meta = pickle.load(f)
                print(f"📂 [VectorStore] Loaded {len(self.meta)} vectors from {self.index_path}")
            except Exception as e:
                print(f"⚠️ [VectorStore] Failed to load existing index: {e}. Starting fresh.")
                self.index = faiss.IndexFlatL2(dim)
                self.meta = []
        else:
            print("🆕 [VectorStore] Creating new FAISS index store...")
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []

    def add_vector(self, vector, meta):
        try:
            v = np.array([vector]).astype("float32")
            if v.shape[1] != self.dim:
                print(f"⚠️ [VectorStore] Vector dim mismatch ({v.shape[1]} vs expected {self.dim}). Auto-fixing...")
                v = v[:, : self.dim] if v.shape[1] > self.dim else np.pad(v, ((0, 0), (0, self.dim - v.shape[1])))

            self.index.add(v)
            self.meta.append(meta)
            self.save()
            print("💾 [VectorStore] Added vector successfully.")
        except Exception as e:
            print("❌ [VectorStore] Error adding vector:", e)

    def search(self, qvec, k=5):
        qv = np.array([qvec]).astype("float32")
        D, I = self.index.search(qv, k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.meta):
                results.append(self.meta[idx])
        return results

    def save(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.meta, f)
            print(f"✅ [VectorStore] Index saved to {self.index_path}")
        except Exception as e:
            print(f"❌ [VectorStore] Failed to save index: {e}")





























# # src/app/storage.py
# import os, pickle
# import numpy as np
# import faiss

# META_FILE = "meta.pkl"

# class VectorStore:
#     def __init__(self, dim=384):
#         self.dim = dim
#         self.index_path = os.getenv("VECTOR_DB_PATH", "/data/faiss_index")
#         if os.path.exists(self.index_path) and os.path.exists(META_FILE):
#             self.index = faiss.read_index(self.index_path)
#             self.meta = pickle.load(open(META_FILE, "rb"))
#         else:
#             self.index = faiss.IndexFlatL2(dim)
#             self.meta = []

#     def add_vector(self, vector, meta):
#         v = np.array([vector]).astype("float32")
#         self.index.add(v)
#         self.meta.append(meta)
#         self.save()

#     def search(self, qvec, k=5):
#         qv = np.array([qvec]).astype("float32")
#         D, I = self.index.search(qv, k)
#         results = []
#         for idx in I[0]:
#             if idx < len(self.meta):
#                 results.append(self.meta[idx])
#         return results

#     def save(self):
#         faiss.write_index(self.index, self.index_path)
#         pickle.dump(self.meta, open(META_FILE, "wb"))
