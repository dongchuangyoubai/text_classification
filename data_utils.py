import numpy as np
import os

class DataLoader(object):
    def __init__(self, input_files):
        self.pos_files = input_files + r"\train\pos\\"
        self.neg_files = input_files + r"\train\neg\\"

    def load_data(self):
        all_pos_files = os.listdir(self.pos_files)
        all_neg_files = os.listdir(self.neg_files)
        all_files = all_pos_files + all_neg_files
        self.labels = []
        self.data = []
        for i in all_pos_files:
            label = i.split('_')[1].split('.')[0]
            self.labels.append(label)
            text = open(self.pos_files + i, 'rb').readline()
            self.data.append(text)

        for i in all_neg_files:
            label = i.split('_')[1].split('.')[0]
            self.labels.append(label)
            text = open(self.neg_files + i, 'rb').readline()
            self.data.append(text)
        #print(self.labels)
        print(len(self.labels))
        print(len(self.data))




if __name__ == '__main__':
    dt = DataLoader(r"D:\python project\text_classification\aclImdb")
    dt.load_data()