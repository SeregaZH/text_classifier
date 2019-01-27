import numpy as np

class MeanEmbeddingsVectorizer(object):
    def __init__(self, w2v):
        self.w2v = w2v
        self.v_size = self.w2v.vector_size

    def vectorize(self, tokens):
        return np.mean(np.array([self.w2v[t] if t in self.w2v.wv.vocab else np.zeros(self.v_size) for t in tokens]), axis=0)
