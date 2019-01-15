import tensorflow as tf
import numpy as np
from data_utils import DataLoader


class Model(object):
    def __init__(self, num_classes, batch_size, seq_lengths, pretrained_embeddings, num_layers=2, learning_rate=0.001,
                 rnn_size=128, dropout=0.5,embedding_size=128):
        self.num_classes = num_classes #10
        self.batch_size = batch_size
        self.embeddings = pretrained_embeddings
        self.seq_lengths = seq_lengths
        self.num_layers = num_layers
        self.lr = learning_rate
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_placeholder()
        self.build_network()
        self.build_loss()
        self.build_optim()
        # self.train()

    def build_placeholder(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, [self.seq_lengths, self.batch_size], 'inputs')
            self.targets = tf.placeholder(tf.int32, [1, self.num_classes], 'targets')
            self.keep_prob = tf.placeholder(tf.float32, name='dropout')
            embeddings = tf.Variable(self.embeddings, trainable=False, name='embeddings', dtype=tf.float32)
            self.inputs_emb = tf.nn.embedding_lookup(embeddings, self.inputs, 'inputs_emb')


    def build_network(self):
        def get_a_cell(rnn_size, dropout):
            lstm = tf.nn.rnn_cell.LSTMCell(rnn_size)
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout)
            return lstm
        
        with tf.name_scope('network'):
            cells = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.rnn_size, self.dropout) for _ in range(self.num_layers)]
            )
            self.initial_state = cells.zero_state(self.seq_lengths, tf.float32)
            self.lstm_output, self.final_state = tf.nn.dynamic_rnn(cells, self.inputs_emb, initial_state=self.initial_state)

            self.lstm_output = tf.concat(self.lstm_output, 1)
            self.lstm_output = tf.reshape(self.lstm_output, [-1, self.rnn_size])

            with tf.variable_scope('softmax'):
                w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
                b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.matmul(self.lstm_output, w) + b
            self.prob = tf.nn.softmax(self.logits)
            self.prob = tf.cast(tf.argmax(self.prob), tf.float32)
            # self.prob = tf.argmax(self.prob)

    def build_loss(self):
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prob, labels=self.targets)
            self.loss = tf.reduce_mean(loss)

    def build_optim(self):
        train_op = tf.train.AdamOptimizer(self.lr)
        self.optimizer = train_op.minimize(self.loss)

    def train(self):
        self.sess = tf.Session()
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.initial_state)



if __name__ == '__main__':
    dt = DataLoader(r"D:\python project\text_classification\aclImdb", 'glove_data', 50, 'trimmed.npz')
    m = Model(10, 32, 128, dt.embeddings['embeddings'])