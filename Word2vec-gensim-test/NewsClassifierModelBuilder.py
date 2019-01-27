import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class NewsClassifierModelBuilder(object):
    def __init__(self, vect_size, label_size):
        self.model = Sequential([
                Dense(64, activation='relu', input_dim=vect_size),
                Dense(label_size, activation='softmax')
            ])

    def compile(self):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    def train(selfl, dataset, labels, num_classes):
        labales_matrix = keras.utils.to_categorical(labels, num_classes)
        self.model.fit(dataset, label_matrix, epochs=20, batch_size=128)