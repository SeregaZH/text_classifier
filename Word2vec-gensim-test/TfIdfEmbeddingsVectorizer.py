from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

class TfIdfEmbeddingsVectorizer:
    def __init__(self, w2v, tokenize):
        self.w2v = w2v
        self.v_size = self.w2v.vector_size
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')

    def vectorize(self, tokens):
        return np.mean([self.w2v[w] * self.word2weight[w]
                         for w in tokens if w in self.w2v] or
                        [np.zeros(self.v_size)], axis=0)

    def fit(self, content):
        self.tfidf.fit(content)
        max_idf = max(self.tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, self.tfidf.idf_[i]) for w, i in self.tfidf.vocabulary_.items()])
        return self
