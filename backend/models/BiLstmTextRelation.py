import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Bidirectional, Dense, GlobalAveragePooling1D
from loss import errors_mean  # Assuming you have your custom loss function defined in loss.py

class BiLstmTextRelation:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, is_training, is_classifier, optimizer,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.optimizer = optimizer

        # add placeholder (X,label)
        self.input_x = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="input_x")
        if is_classifier:
            self.input_y = tf.keras.Input(shape=(), dtype=tf.int32, name="input_y")
        else:
            self.input_y = tf.keras.Input(shape=(num_classes,), dtype=tf.float32, name="input_y")

        self.dropout_keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        if optimizer == 'Adam':
            self.learning_rate = 0.001

        self.instantiate_weights()
        # [None, self.label_size]. main computation graph is here.
        self.logits = self.inference()
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        if self.is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            # self.accuracy = tf.constant(0.5)
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction)

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)

        self.W_projection = tf.Variable(tf.random.truncated_normal((self.hidden_size * 2, self.num_classes), stddev=0.1))
        self.b_projection = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

    def inference(self):
        """
        main computation graph here:
        1. embedding layer, 2. Bi-LSTM layer,
        3. mean pooling, 4. FC layer, 5. softmax
        """

        # 1. get embedding of words in the sentence
        self.embedded_words = self.Embedding(self.input_x)

        # 2. Bi-lstm layer
        lstm_fw_cell = LSTM(self.hidden_size, return_sequences=True)
        lstm_bw_cell = LSTM(self.hidden_size, return_sequences=True)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = Dropout(rate=1 - self.dropout_keep_prob)(lstm_fw_cell)
            lstm_bw_cell = Dropout(rate=1 - self.dropout_keep_prob)(lstm_bw_cell)

        outputs = Bidirectional(layer=lstm_fw_cell, backward_layer=lstm_bw_cell)(self.embedded_words)

        # 3. mean pooling
        output_rnn_pooled = GlobalAveragePooling1D()(outputs)

        # 4. logits(use linear layer)
        logits = tf.matmul(output_rnn_pooled, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):
        if self.is_classifier:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                  if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        else:
            loss = errors_mean(self.input_y, self.logits)
        return loss

    def train(self):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.global_step,
                                                                       self.decay_steps, self.decay_rate, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer = optimizer
        train_op = optimizer.minimize(self.loss_val, global_step=self.global_step)
        return train_op
