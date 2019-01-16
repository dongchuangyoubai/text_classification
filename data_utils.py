import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from tqdm import *
import tensorflow as tf
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_generator(inputs, targets, batch_size, seq_length):
    n_batch = int(len(inputs)/batch_size)
    inputs = inputs[:n_batch * batch_size]
    targets = targets[:n_batch * batch_size]
    while True:
        for i in range(0, len(inputs), batch_size):
            x = inputs[i: i + batch_size]
            x_padd = []
            for j in x:
                if len(j) >= seq_length:
                    x_padd.append(j[:seq_length])
                else:
                    x_padd.append(j+[4999 for _ in range(seq_length - len(j))])
            y = targets[i: i + batch_size]
            # y = [tf.one_hot(int(i), 10) for i in y]
            yield x_padd, y


class DataLoader(object):
    def __init__(self, input_files, embed_dir, embed_size, npz_path, pkl_file='data.pkl', max_vocab_size=5000):
        self.pos_files = input_files + r"\train\pos\\"
        self.neg_files = input_files + r"\train\neg\\"
        self.embed_size = embed_size
        self.npz_path = npz_path
        self.embed_path = os.path.join(embed_dir, "glove.6B.{}d.txt".format(embed_size))
        self.pkl_file = pkl_file
        self.max_vocab_size = max_vocab_size
        self.max_seq_leg = 0
        self.load_data()
        self.load_emb(4e5)

    def load_data(self):
        if os.path.exists(self.pkl_file):
            logger.info("load data from %s" % self.pkl_file)
            [self.data, self.labels, self.vocab_list, self.word2int_dic, self.int2word_dic, self.max_seq_leg] = \
                pickle.load(open(self.pkl_file, 'rb'))

        else:
            logger.info("create pkl to %s" % self.pkl_file)
            english_stopwords = stopwords.words('english')
            english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', "/"
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
                # text_stemmed = [st.stem(w) for w in text_filter_punc]
                self.data.append(text_filter_punc)
                if len(text_filter_punc) > self.max_seq_leg:
                    self.max_seq_leg = len(text_filter_punc)

            for i in all_neg_files:
                label = i.split('_')[1].split('.')[0]
                self.labels.append(label)
                text = open(self.neg_files + i, 'r', encoding='utf-8').readline()
                text_tokenized = [w for w in nltk.word_tokenize(text)]
                text_filter_stopwords = [w for w in text_tokenized if w not in english_stopwords]
                text_filter_punc = [w for w in text_filter_stopwords if w not in english_punctuations]
                # text_stemmed = [st.stem(w) for w in text_filter_punc]
                self.data.append(text_filter_punc)
                if len(text_filter_punc) > self.max_seq_leg:
                    self.max_seq_leg = len(text_filter_punc)

            count_dict = {}
            all_data = [w for l in self.data for w in l]
            for i in all_data:
                if i not in count_dict.keys():
                    count_dict[i] = 0
                count_dict[i] += 1
            self.vocab_list = sorted(count_dict.items(), key=lambda d: d[1], reverse=True)
            self.vocab_list = [w[0] for w in self.vocab_list]
            self.vocab_list = self.vocab_list if self.max_vocab_size > len(self.vocab_list) \
                                              else self.vocab_list[3: 3 + self.max_vocab_size - 1]
            self.vocab_list.append('unk')
            self.word2int_dic = {v: k for k, v in enumerate(self.vocab_list)}
            self.int2word_dic = dict(enumerate(self.vocab_list))
            self.data = [self.text2arr(i) for i in self.data]

            pickle.dump([self.data, self.labels, self.vocab_list, self.word2int_dic,
                         self.int2word_dic, self.max_seq_leg], open(self.pkl_file, 'wb'))


    def load_emb(self, size):
        if os.path.exists(self.npz_path):
            logger.info("load embeddings from %s" % self.npz_path)
            self.embeddings = np.load(self.npz_path)
        else:
            logger.info("create embeddings to %s" % self.npz_path)
            glove = np.random.randn(len(self.vocab_list), self.embed_size)
            found = 0
            print(glove.shape)
            with open(self.embed_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=size):
                    array = line.lstrip().rstrip().split(" ")
                    word = array[0]
                    vector = list(map(float, array[1:]))
                    if word in self.vocab_list:
                        idx = self.vocab_list.index(word)
                        glove[idx, :] = vector
                        found += 1
            self.embeddings = glove
            print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(self.vocab_list), self.embed_path))
            np.savez_compressed(self.npz_path, embeddings=self.embeddings)
            print("saved trimmed glove matrix at: {}".format(self.npz_path))

    def word2int(self, word):
        if word in self.vocab_list:
            return self.word2int_dic[word]
        else:
            return len(self.vocab_list) - 1

    def int2word(self, idx):
        return self.int2word_dic[idx]

    def text2arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word2int(word))
        return arr

    def arr2text(self, arr):
        text = []
        for i in arr:
            text.append(self.int2word(i))
        return ''.join(text)


if __name__ == '__main__':
    dt = DataLoader(r"D:\python project\text_classification\aclImdb", 'glove_data', 50, 'trimmed.npz')
    for x, y in batch_generator(dt.data, dt.labels, 32, 200):
        for i in x:
            print(i)

        break

