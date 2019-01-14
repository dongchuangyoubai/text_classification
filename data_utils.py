import numpy as np
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora

class DataLoader(object):
    def __init__(self, input_files, pkl_file='data.pkl', max_vocab_size):
        self.pos_files = input_files + r"\train\pos\\"
        self.neg_files = input_files + r"\train\neg\\"
        self.pkl_file = pkl_file
        self.load_data()
        self.build_dict()

    def load_data(self):
        if os.path.exists(self.pkl_file):
            [self.data, self.labels] = pickle.load(open(self.pkl_file, 'rb'))
            print(len(self.labels))
        else:
            english_stopwords = stopwords.words('english')
            english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']',
                                    '&', '!', '*', '@', '#', '$', '%', '-', '``', "''"]
            st = LancasterStemmer()
            all_pos_files = os.listdir(self.pos_files)
            all_neg_files = os.listdir(self.neg_files)
            self.labels = []
            self.data = []
            self.vocab = []
            for i in all_pos_files:
                label = i.split('_')[1].split('.')[0]
                self.labels.append(label)
                text = open(self.pos_files + i, 'r', encoding='utf-8').readline()
                text_tokenized = [w for w in nltk.word_tokenize(text)]
                text_filter_stopwords = [w for w in text_tokenized if w not in english_stopwords]
                text_filter_punc = [w for w in text_filter_stopwords if w not in english_punctuations]
                text_stemmed = [st.stem(w) for w in text_filter_punc]
                self.data.append(text_stemmed)
                # print(text)
                # print(text_tokenized)
                # print(text_filter_stopwords)
                # print(text_filter_punc)
                # print(text_stemmed)
                # break

            for i in all_neg_files:
                label = i.split('_')[1].split('.')[0]
                self.labels.append(label)
                text = open(self.neg_files + i, 'r', encoding='utf-8').readline()
                text_tokenized = [w for w in nltk.word_tokenize(text)]
                text_filter_stopwords = [w for w in text_tokenized if w not in english_stopwords]
                text_filter_punc = [w for w in text_filter_stopwords if w not in english_punctuations]
                text_stemmed = [st.stem(w) for w in text_filter_punc]
                self.data.append(text_stemmed)
                # print(text)
                # print(text_tokenized)
                # print(text_filter_stopwords)
                # print(text_filter_punc)
                # print(text_stemmed)
                # break

            pickle.dump([self.data, self.labels], open('data.pkl', 'wb'))

    def build_dict(self):
        self.count_dict = {}
        all_data = [w for l in self.data for w in l]
        print(len(all_data))
        print(all_data[:10])
        self.vocab = set(all_data)
        print(len(self.vocab))
        self.




if __name__ == '__main__':
    dt = DataLoader(r"D:\python project\text_classification\aclImdb")
    # dt = DataLoader(r"D:\python project\text_classification\aclImdb")
