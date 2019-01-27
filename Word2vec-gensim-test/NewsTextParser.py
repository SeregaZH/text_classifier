import os
import numpy as np
from os.path import isfile, join
import nltk
# nltk.download()

class NewsTextParser(object):
    def __init__(self, base_path):
        self.base_path = base_path

    def __read_content(self, file_path):
        with open(file_path, 'r') as file:
            text=file.read().replace('\n', '')
            print('File by path {0} readed'.format(file_path))
            return nltk.word_tokenize(text)

    def load_dataset(self):
       dirs = [dir for dir in os.listdir(self.base_path)]
       files_dirs = [[d, [fls for fls in os.listdir(join(self.base_path, d)) if isfile(join(self.base_path, d, fls))]] for d in dirs ]
       dirs_content = [[fd[0], np.array([np.array(self.__read_content(join(self.base_path, fd[0], f))) for f in fd[1]])] for fd in files_dirs]
       return dirs_content, self.create_labels(dirs)

    def create_labels(self, dirs):
       dict = {}
       for idx, val in enumerate(dirs):
           dict[val] = idx
       return dict