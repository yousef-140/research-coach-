import faiss
import numpy as np
import os 


class FaissIndex:
    def __init__(self, dimension: int, path: str = 'faiss_index.bin'):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.path = path
        if os.path.exists(path):
            self.index = faiss.read_index(path)

    def add(self , vectors:np.ndarray):
        self.index.add(vectors)
        self.save()


    def save(self):
        faiss.write_index(self.index ,self.path)
        

    def search(self , query_vector:np.ndarray, top_k:int):
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices