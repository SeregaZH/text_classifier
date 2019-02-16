import os
import numpy as np
from os.path import isfile, join
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# nltk.download()

class Tokenizer():
    def __init__(self, stemmer = nltk.PorterStemmer(), tockenizer = RegexpTokenizer(r'\w+'), stopW = set(stopwords.words('english'))):
        self.stemmer = stemmer
        self.tockenizer = tockenizer
        self.stopW = stopW

    def tokenize(self, text):
        return [self.stemmer.stem(w).lower() for w in self.tockenizer.tokenize(text) if w.lower() not in self.stopW]

default_tokenizer = Tokenizer()

class NewsTextParser(object):
    def __init__(self, base_path):
        self.base_path = base_path
        self.tockenizer = default_tokenizer

    def __read_content(self, file_path):
        with open(file_path, 'r') as file:
            text=file.read().replace('\n', '')
            print('File by path {0} readed'.format(file_path))
            return text

    def load_dataset(self):
       dirs = [dir for dir in os.listdir(self.base_path)]
       files_dirs = np.vstack(np.array([np.array([(d, fls) for fls in os.listdir(join(self.base_path, d)) if isfile(join(self.base_path, d, fls))]) for d in dirs ]))
       dirs_content = [(fd[0], self.__read_content(join(self.base_path, fd[0], fd[1]))) for fd in files_dirs]
       dirs_tokens = [ [content[0], self.tockenizer.tokenize(content[1])] for content in dirs_content]
       return dirs_tokens, self.create_labels(dirs), dirs_content

    def create_labels(self, dirs):
       dict = {}
       for idx, val in enumerate(dirs):
           dict[val] = idx
       return dict