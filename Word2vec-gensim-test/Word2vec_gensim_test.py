import os
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from MeanEmbeddingsVectorizer import MeanEmbeddingsVectorizer
import numpy as np
import argparse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from NewsTextParser import NewsTextParser
from NewsClassifierModelBuilder import NewsClassifierModelBuilder

plotly.tools.set_credentials_file(username='seregazh', api_key='************')

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

# hardcoded params
epoch = 200

text_parser = NewsTextParser(full_path)
dataset, labels = text_parser.load_dataset()
examples_list = flatten_one([[ (ex, examples[0]) for ex in examples[1]] for examples in dataset])
dataset_list = np.array([[i[0].tolist(), labels[i[1]]] for i in examples_list])
np.random.shuffle(dataset_list)
training_text_list, labels_list = (dataset_list[:,0], dataset_list[:,1]) 

if not os.path.exists(model_path) or args.train:
    model = Word2Vec(training_text_list, size=100, window=5, min_count=1, workers=4)
    model.save(model_path)
else:
    model = Word2Vec.load(model_path)

news_model = NewsClassifierModelBuilder(model.wv.vector_size, len(labels))
news_model.compile()
mean_vectorizer = MeanEmbeddingsVectorizer(model)
mean_vectors = [mean_vectorizer.vectorize(v) for v in training_text_list]
history = news_model.train(np.array(mean_vectors), labels_list, len(labels), epoch)

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