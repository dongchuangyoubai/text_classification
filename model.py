import tensorflow as tf
import numpy as np
from data_utils import DataLoader, batch_generator
import logging
logger = logging.getLogger(__name__)

class Model(object):
    def __init__(self, data_loader, num_classes, batch_size, seq_lengths, num_layers=1, learning_rate=0.1,
                 hidden_size=128, dropout=0.5, grad_clip=5):
        self.dt = data_loader
        self.num_classes = num_classes #10
        self.batch_size = batch_size
        self.embeddings = self.dt.embeddings['embeddings']
        self.seq_lengths = seq_lengths
        self.num_layers = num_layers
        self.lr = learning_rate
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.grad_clip = grad_clip

        tf.reset_default_graph()
        self.build_placeholder()
        self.build_network()
        self.build_loss()
        self.build_optim()
        # self.train()

    def build_placeholder(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_lengths], 'inputs')
            self.targets = tf.placeholder(tf.int64, [self.batch_size], 'targets')
            embeddings = tf.Variable(self.embeddings, trainable=False, name='embeddings', dtype=tf.float32)
            self.inputs_emb = tf.nn.embedding_lookup(embeddings, self.inputs, 'inputs_emb')

    def build_network(self):
        def get_a_cell(hidden_size, dropout):
            lstm = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout)
            return lstm
        
        with tf.name_scope('network'):
            cells = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.hidden_size, self.dropout) for _ in range(self.num_layers)]
            )
            self.initial_state = cells.zero_state(self.batch_size, tf.float32)
            self.lstm_output, self.final_state = tf.nn.dynamic_rnn(cells, self.inputs_emb,
                                                                   initial_state=self.initial_state)

            c, h = self.final_state[0]
            # self.lstm_output = tf.concat(self.final_state, 1)
            # self.lstm_output = tf.reshape(self.final_state, (-1, self.hidden_size))

            with tf.variable_scope('softmax'):
                w = tf.Variable(tf.truncated_normal([self.hidden_size, self.num_classes], mean=0, stddev=0.1))
                b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.matmul(h, w) + b
            self.prob = tf.map_fn(lambda i: tf.nn.softmax(i), self.logits)
            self.prob1 = tf.argmax(self.prob, axis=1)

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_one_hot)
            self.loss = tf.reduce_mean(loss)

    def build_optim(self):
        # train_op = tf.train.AdamOptimizer(self.lr)
        # self.optimizer = train_op.minimize(self.loss)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.lr)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self):
        logger.info("start train")
        self.sess = tf.Session()
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            new_state = sess.run(self.initial_state)
            avg_loss = 0
            for x, y in batch_generator(self.dt.data, self.dt.labels, self.batch_size, self.seq_lengths):
                feed = {self.inputs: x,
                            self.targets: y,
                            self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer],
                                                    feed_dict=feed)
                step += 1
                avg_loss += batch_loss
                if step % 100 == 0:
                    print("steps: %d, batch_loss: %f"% (step, avg_loss/100))
                    avg_loss = 0
                    correct_prediction = tf.equal(self.prob1, self.targets)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("Accuracy:", accuracy.eval({self.inputs: x, self.targets: y}))
                    print("targets", self.targets.eval({self.inputs: x, self.targets: y}))
                    print("logits", self.logits.eval({self.inputs: x, self.targets: y}))
                    print("prob", self.prob.eval({self.inputs: x, self.targets: y}))
                    print("prob1", self.prob1.eval({self.inputs: x, self.targets: y}))
                if step == 7810:
                    break


if __name__ == '__main__':
    dt = DataLoader(r"D:\python project\text_classification\aclImdb", 'glove_data', 50, 'trimmed.npz')
    m = Model(dt, 10, 100, 200)
    m.train()