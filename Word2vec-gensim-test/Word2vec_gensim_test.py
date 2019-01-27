import os
from MeanEmbeddingsVectorizer import MeanEmbeddingsVectorizer
import numpy as np
import argparse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from NewsTextParser import NewsTextParser
from NewsClassifierModelBuilder import NewsClassifierModelBuilder

def flatten_one(arr):
    result = []
    for l1 in arr:
        for i in l1:
          result.append(i)
    return result

# extract arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true')
args = parser.parse_args()

# set default paths
current_dir = os.path.dirname(__file__)
model_name = 'word2vec.model'
temp_path = os.path.join(current_dir, 'temp')
test_path = os.path.join(current_dir, 'test')
full_path = os.path.join(current_dir, 'dataset')
model_path = os.path.join(current_dir, model_name)

text_parser = NewsTextParser(test_path)
dataset, labels = text_parser.load_dataset()
examples_list = flatten_one([[ (ex, examples[0]) for ex in examples[1]] for examples in dataset])
training_text_list = [i[0].tolist() for i in examples_list]
labels_list = [labels[i[1]] for i in examples_list]

if not os.path.exists(model_path) or args.train:
    model = Word2Vec(training_text_list, size=100, window=5, min_count=1, workers=4)
    model.save(model_path)
else:
    model = Word2Vec.load(model_path)

news_model = NewsClassifierModelBuilder(model.wv.vector_size, len(labels))
train_parser = NewsTextParser(test_path)
mean_vectorizer = MeanEmbeddingsVectorizer(model)
mean_vector = mean_vectorizer.vectorize(dataset[0][1][1])
print(mean_vector)

