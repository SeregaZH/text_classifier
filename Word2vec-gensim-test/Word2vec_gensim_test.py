import os
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pickle as pkl
from TfIdfEmbeddingsVectorizer import TfIdfEmbeddingsVectorizer
import numpy as np
import argparse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from NewsTextParser import NewsTextParser, Tokenizer
from NewsClassifierModelBuilder import NewsClassifierModelBuilder

plotly.tools.set_credentials_file(username='seregazh', api_key='******************')

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
vectors_file = 'vectorfile.pkl'

# hardcoded params
epoch = 400

if not os.path.exists(model_path) or args.train:
    text_parser = NewsTextParser(full_path)
    dataset, labels, content = text_parser.load_dataset()
    dataset_list = np.array([[i[1], labels[i[0]]] for i in dataset])
    content_list = [c[1] for c in content]
    np.random.shuffle(dataset_list)
    training_text_list, labels_list = (dataset_list[:,0], dataset_list[:,1])
    model = Word2Vec(training_text_list, size=100, window=5, min_count=1, workers=8)
    model.save(model_path)
    tfidf_vectorizer = TfIdfEmbeddingsVectorizer(model, Tokenizer().tokenize)
    tfidf_vectorizer.fit(content_list)
    doc_vectors = np.array([tfidf_vectorizer.vectorize(v) for v in training_text_list])
    with open(vectors_file, "wb") as vfile:
        pkl.dump((doc_vectors, labels_list, labels), vfile)
else:
    model = Word2Vec.load(model_path)
    with open(vectors_file, "rb") as vfile:
        data = pkl.load(vfile)
        doc_vectors, labels_list, labels = data

news_model = NewsClassifierModelBuilder(model.wv.vector_size, len(labels))
news_model.compile()
history = news_model.train(np.array(doc_vectors), labels_list, len(labels), epoch)

# create chart
x_dim = np.linspace(0, epoch, epoch, dtype = np.int32)
loss = go.Scatter(
    x = x_dim,
    y = history.history['loss'],
    mode = 'lines',
    name = 'loss'
)
accuracy = go.Scatter(
    x = x_dim,
    y = history.history['acc'],
    mode = 'lines',
    name = 'accuracy'
)
val_loss = go.Scatter(
    x = x_dim,
    y = history.history['val_loss'],
    mode = 'lines',
    name = 'cross-validation loss'
)
val_accuracy = go.Scatter(
    x = x_dim,
    y = history.history['val_acc'],
    mode = 'lines',
    name = 'cross-validation accuracy'
)
data = [loss, accuracy, val_loss, val_accuracy]
py.plot(data, filename='line-mode')