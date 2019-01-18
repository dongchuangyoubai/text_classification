import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import re
from random import randint
import datetime

wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')

print(len(wordsList))
print(wordsList[:10])
print(wordVectors.shape)

positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
# numWords = []
# for pf in positiveFiles:
#     with open(pf, "r", encoding='utf-8') as f:
#         line = f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Positive files finished')
#
# for nf in negativeFiles:
#     with open(nf, "r", encoding='utf-8') as f:
#         line = f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Negative files finished')
#
# numFiles = len(numWords)
# print('The total number of files is', numFiles)
# print('The total number of words in the files is', sum(numWords))
# print('The average number of words in the files is', sum(numWords)/len(numWords))


Max_Seq_Length = 250

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

# numFiles = 25000
# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
#    with open(pf, "r", encoding='utf-8') as f:
#        indexCounter = 0
#        line = f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
#
# for nf in negativeFiles:
#    with open(nf, "r", encoding='utf-8') as f:
#        indexCounter = 0
#        line = f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
# #Pass into embedding function and see if it evaluates.
#
# np.save('idsMatrix', ids)

ids = np.load('idsMatrix.npy')

Batch_Size = 32
Hidden_Units = 128
Num_Classes = 2
Iterations = 100000

def get_train_batch():
    labels = []
    arr = np.zeros([Batch_Size, Max_Seq_Length])
    for i in range(Batch_Size):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def get_test_batch():
    labels = []
    arr = np.zeros([Batch_Size, Max_Seq_Length])
    for i in range(Batch_Size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

tf.reset_default_graph()

inputs = tf.placeholder(tf.int32, [Batch_Size, Max_Seq_Length])
targets = tf.placeholder(tf.int32, [Batch_Size, Num_Classes])
inputs_emb = tf.nn.embedding_lookup(wordVectors, inputs)
cell = tf.nn.rnn_cell.LSTMCell(Hidden_Units)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.75)
output, [c, h] = tf.nn.dynamic_rnn(cell, inputs_emb, dtype=tf.float32)
# print(output.shape.as_list())
output = tf.transpose(output, [1, 0, 2])
# print(int(output.get_shape()[0]) - 1)
output = tf.gather(output, int(output.get_shape()[0]) - 1)
# print(a.shape.as_list())

w = tf.Variable(tf.truncated_normal([Hidden_Units, Num_Classes]))
b = tf.Variable(tf.constant(0.1, shape=[Num_Classes]))

pred = tf.matmul(output, w) + b

correct = tf.equal(tf.argmax(pred, 1), tf.argmax(targets, 1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', acc)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(Iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = get_train_batch()
   sess.run(optimizer, {inputs: nextBatch, targets: nextBatchLabels})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {inputs: nextBatch, targets: nextBatchLabels})
       writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)
writer.close()





