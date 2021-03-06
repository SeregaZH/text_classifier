import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

class NewsClassifierModelBuilder(object):
    def __init__(self, vect_size, label_size):
        self.model = Sequential([
                Dense(100, activation='relu', input_dim=vect_size),
                Dropout(0.1),
                Dense(200, activation='relu'),
                Dropout(0.3),
                Dense(100, activation='relu'),
                Dropout(0.1),
                Dense(label_size, activation='softmax')
            ])

    def compile(self):
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    def train(self, dataset, labels, num_classes, epochs=20, batch_size=64):
        labales_matrix = keras.utils.to_categorical(labels, num_classes)
        return self.model.fit(dataset, labales_matrix, epochs=epochs, batch_size=batch_size, validation_split=0.2)