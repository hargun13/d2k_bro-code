# import numpy as np
# import tensorflow as tf
# # Example usage of SimpleRNN in TensorFlow 2.x
# from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense
#
#
# from loss import errors_mean
#
# class TextRNN:
#     def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
#                  sequence_length, vocab_size, embed_size, is_training, batchnorm, is_classifier,
#                  optimizer, initializer=tf.random_normal_initializer(stddev=0.1)):
#         # Initialize hyperparameters
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.sequence_length = sequence_length
#         self.vocab_size = vocab_size
#         self.embed_size = embed_size
#         self.hidden_size = embed_size
#         self.is_training = is_training
#         self.learning_rate = learning_rate
#         self.initializer = initializer
#         self.batchnorm = batchnorm
#         self.num_sampled = 20
#         self.is_classifier = is_classifier
#         self.optimizer = optimizer
#
#
#
#         # Define placeholders
#         self.input_x = tf.keras.Input(shape=(self.sequence_length,), dtype=tf.int32, name="input_x")
#         if is_classifier:
#             self.input_y = tf.keras.Input(shape=(), dtype=tf.int32, name="input_y")
#         else:
#             self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
#         self.dropout_keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name="dropout_keep_prob")
#
#         self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
#         self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
#         self.epoch_increment = self.epoch_step.assign_add(1)
#         self.decay_steps, self.decay_rate = decay_steps, decay_rate
#
#         self.instantiate_weights()
#         self.logits = self.inference()
#
#         if not is_training:
#             return
#         self.loss_val = self.loss()
#         self.train_op = self.train()
#         self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
#
#         correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
#         if is_classifier:
#             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
#         else:
#             self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction, name="Accuracy")
#
#     def instantiate_weights(self):
#         with tf.name_scope("embedding"):
#             self.Embedding = tf.Variable(tf.random.normal([self.vocab_size, self.embed_size]), name="Embedding")
#             self.W_projection = tf.Variable(tf.random.normal([self.hidden_size * 2, self.num_classes]),
#                                             name="W_projection")
#             self.b_projection = tf.Variable(tf.random.normal([self.num_classes]), name="b_projection")
#
#     def inference(self):
#         self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
#
#         lstm_fw_cell = LSTM(self.hidden_size, return_sequences=True)
#         lstm_bw_cell = LSTM(self.hidden_size, return_sequences=True)
#         if self.dropout_keep_prob is not None:
#             # Include dropout wrappers here if necessary
#             pass
#
#         outputs = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(self.embedded_words)
#
#         output_rnn = tf.concat(outputs, axis=2)
#         self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
#
#         with tf.name_scope("output"):
#             logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
#         return logits
#     def loss(self, l2_lambda=0.0001):
#         with tf.name_scope("loss"):
#             if self.is_classifier:
#                 losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
#                                                                         logits=self.logits)
#                 loss = tf.reduce_mean(losses)
#                 import tensorflow.compat.v1 as tf_compat
#                 l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf_compat.trainable_variables()])
#                 loss = loss + l2_losses
#             else:
#                 loss = errors_mean(self.input_y, self.logits)
#         return loss
#
#     def train(self):
#         learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
#                                                    self.decay_rate, staircase=True)
#         train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
#                                                    learning_rate=learning_rate, optimizer=self.optimizer)
#         return train_op
#
#     def batch_norm(self, x, n_out, phase_train):
#         with tf.variable_scope('bn'):
#             beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
#             gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
#             batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
#             ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#             def mean_var_with_update():
#                 ema_apply_op = ema.apply([batch_mean, batch_var])
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)
#
#             mean, var = tf.cond(phase_train, mean_var_with_update,
#                                 lambda: (ema.average(batch_mean), ema.average(batch_var)))
#             bnormed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#         return bnormed
#
# # Define filter_sizes and num_filters
# filter_sizes = [3, 4, 5]
# num_filters = 128
#
# # Test phase
# def test():
#     num_classes = 49
#     learning_rate = 0.01
#     batch_size = 8
#     decay_steps = 1000
#     decay_rate = 0.9
#     sequence_length = 5
#     vocab_size = 10000
#     embed_size = 100
#     is_training = True
#     dropout_keep_prob = 1
#     batchnorm = True
#     is_classifier = True
#     optimizer = 'Adam'  # or any other optimizer you want to use
#
#     # Make sure you pass all required arguments when creating an instance of TextRNN
#     textRNN = TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, vocab_size,
#                       embed_size, is_training, batchnorm, is_classifier, optimizer)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(100):
#             input_x = np.zeros((batch_size, sequence_length))
#             input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1])
#             loss, acc, predict, _ = sess.run(
#                 [textRNN.loss_val, textRNN.accuracy, textRNN.predictions, textRNN.train_op],
#                 feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y,
#                            textRNN.dropout_keep_prob: dropout_keep_prob})
#             print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
#
# test()

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import LSTM, Bidirectional
#
# class TextRNN:
#     def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
#                  sequence_length, vocab_size, embed_size, is_training, batchnorm, is_classifier,
#                  optimizer, initializer=tf.random_normal_initializer(stddev=0.1)):
#         # Initialize hyperparameters
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.sequence_length = sequence_length
#         self.vocab_size = vocab_size
#         self.embed_size = embed_size
#         self.hidden_size = embed_size
#         self.is_training = is_training
#         self.learning_rate = learning_rate
#         self.initializer = initializer
#         self.batchnorm = batchnorm
#         self.num_sampled = 20
#         self.is_classifier = is_classifier
#         self.optimizer = optimizer
#
#         # Define placeholders
#         self.input_x = tf.keras.Input(shape=(self.sequence_length,), dtype=tf.int32, name="input_x")
#         if is_classifier:
#             self.input_y = tf.keras.Input(shape=(), dtype=tf.int32, name="input_y")
#         else:
#             self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
#         self.dropout_keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name="dropout_keep_prob")
#
#         self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
#         self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
#         self.epoch_increment = self.epoch_step.assign_add(1)
#         self.decay_steps, self.decay_rate = decay_steps, decay_rate
#
#         self.instantiate_weights()
#         self.logits = self.inference()
#
#         if not is_training:
#             return
#         self.loss_val = self.loss()
#         self.train_op = self.train()
#         self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
#
#         correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
#         if is_classifier:
#             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
#         else:
#             self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction, name="Accuracy")
#
#     def instantiate_weights(self):
#         with tf.name_scope("embedding"):
#             self.Embedding = tf.Variable(tf.random.normal([self.vocab_size, self.embed_size]), name="Embedding")
#             self.W_projection = tf.Variable(tf.random.normal([self.hidden_size * 2, self.num_classes]),
#                                             name="W_projection")
#             self.b_projection = tf.Variable(tf.random.normal([self.num_classes]), name="b_projection")
#
#     def inference(self):
#         self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
#
#         lstm_fw_cell = LSTM(self.hidden_size, return_sequences=True)
#         lstm_bw_cell = LSTM(self.hidden_size, return_sequences=True)
#
#         outputs = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(self.embedded_words)
#
#         output_rnn = tf.concat(outputs, axis=2)
#         self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
#
#         with tf.name_scope("output"):
#             logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
#         return logits
#
#     def loss(self, l2_lambda=0.0001):
#         with tf.name_scope("loss"):
#             if self.is_classifier:
#                 losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
#                 loss = tf.reduce_mean(losses)
#                 # Get only the variables you want to include in the L2 loss calculation
#                 l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
#                                       if 'W_projection' in v.name or 'Embedding' in v.name])
#                 loss = loss + l2_lambda * l2_losses
#             else:
#                 loss = errors_mean(self.input_y, self.logits)
#         return loss
#     def train(self):
#         learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
#                                                    self.decay_rate, staircase=True)
#         train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss_val, global_step=self.global_step)
#         return train_op
#
# # Define filter_sizes and num_filters
# filter_sizes = [3, 4, 5]
# num_filters = 128
#
# # Test phase
# def test():
#     num_classes = 49
#     learning_rate = 0.01
#     batch_size = 8
#     decay_steps = 1000
#     decay_rate = 0.9
#     sequence_length = 5
#     vocab_size = 10000
#     embed_size = 100
#     is_training = True
#     dropout_keep_prob = 1
#     batchnorm = True
#     is_classifier = True
#     optimizer = 'Adam'  # or any other optimizer you want to use
#
#     # Make sure you pass all required arguments when creating an instance of TextRNN
#     textRNN = TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, vocab_size,
#                       embed_size, is_training, batchnorm, is_classifier, optimizer)
#
#     with tf.compat.v1.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())
#         for i in range(100):
#             input_x = np.zeros((batch_size, sequence_length))
#             input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1])
#             loss, acc, predict, _ = sess.run(
#                 [textRNN.loss_val, textRNN.accuracy, textRNN.predictions, textRNN.train_op],
#                 feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y,
#                            textRNN.dropout_keep_prob: dropout_keep_prob})
#             print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
#
# test()

#
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import LSTM, Bidirectional
#
#
# class TextRNN:
#     def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
#                  sequence_length, vocab_size, embed_size, is_training, batchnorm, is_classifier,
#                  optimizer, initializer=tf.random_normal_initializer(stddev=0.1)):
#         # Initialize hyperparameters
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.sequence_length = sequence_length
#         self.vocab_size = vocab_size
#         self.embed_size = embed_size
#         self.hidden_size = embed_size
#         self.is_training = is_training
#         self.learning_rate = learning_rate
#         self.initializer = initializer
#         self.batchnorm = batchnorm
#         self.num_sampled = 20
#         self.is_classifier = is_classifier
#         self.optimizer = optimizer
#
#         # Define placeholders
#         self.input_x = tf.keras.Input(shape=(self.sequence_length,), dtype=tf.int32, name="input_x")
#         if is_classifier:
#             self.input_y = tf.keras.Input(shape=(), dtype=tf.int32, name="input_y")
#         else:
#             self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
#         self.dropout_keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name="dropout_keep_prob")
#
#         self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
#         self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
#         self.epoch_increment = self.epoch_step.assign_add(1)
#         self.decay_steps, self.decay_rate = decay_steps, decay_rate
#
#         self.instantiate_weights()
#         self.logits = self.inference()
#
#         if not is_training:
#             return
#         self.loss_val = self.loss()
#         self.train_op = self.train()
#         self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
#
#         correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
#         if is_classifier:
#             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
#         else:
#             self.accuracy = tf.compat.v1.metrics.mean_squared_error(self.input_y, correct_prediction, name="Accuracy")
#
#     def instantiate_weights(self):
#         with tf.name_scope("embedding"):
#             self.Embedding = tf.Variable(tf.random.normal([self.vocab_size, self.embed_size]), name="Embedding")
#             self.W_projection = tf.Variable(tf.random.normal([self.hidden_size * 2, self.num_classes]),
#                                             name="W_projection")
#             self.b_projection = tf.Variable(tf.random.normal([self.num_classes]), name="b_projection")
#
#     def inference(self):
#         self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
#
#         lstm_fw_cell = LSTM(self.hidden_size, return_sequences=True)
#         lstm_bw_cell = LSTM(self.hidden_size, return_sequences=True)
#
#         outputs = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(self.embedded_words)
#
#         output_rnn = tf.concat(outputs, axis=2)
#         self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1)
#
#         with tf.name_scope("output"):
#             logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
#         return logits
#
#     def loss(self, l2_lambda=0.0001):
#         with tf.name_scope("loss"):
#             if self.is_classifier:
#                 losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
#                 loss = tf.reduce_mean(losses)
#                 # Get only the variables you want to include in the L2 loss calculation
#                 l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()
#                                       if 'W_projection' in v.name or 'Embedding' in v.name])
#                 loss = loss + l2_lambda * l2_losses
#             else:
#                 loss = errors_mean(self.input_y, self.logits)
#         return loss
#
#     def train(self):
#         learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
#                                                              self.decay_rate, staircase=True)
#         train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss_val, global_step=self.global_step)
#         return train_op
#
#
# # Define filter_sizes and num_filters
# filter_sizes = [3, 4, 5]
# num_filters = 128
#
#
# # Test phase
# def test():
#     num_classes = 49
#     learning_rate = 0.01
#     batch_size = 8
#     decay_steps = 1000
#     decay_rate = 0.9
#     sequence_length = 5
#     vocab_size = 10000
#     embed_size = 100
#     is_training = True
#     dropout_keep_prob = 1
#     batchnorm = True
#     is_classifier = True
#     optimizer = 'Adam'  # or any other optimizer you want to use
#
#     # Make sure you pass all required arguments when creating an instance of TextRNN
#     textRNN = TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, vocab_size,
#                       embed_size, is_training, batchnorm, is_classifier, optimizer)
#
#     with tf.compat.v1.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())
#         for i in range(100):
#             input_x = np.zeros((batch_size, sequence_length))
#             input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1])
#             loss, acc, predict, _ = sess.run(
#                 [textRNN.loss_val, textRNN.accuracy, textRNN.predictions, textRNN.train_op],
#                 feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y,
#                            textRNN.dropout_keep_prob: dropout_keep_prob})
#             print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
#
#
# test()




import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional

class TextRNN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size, is_training, batchnorm, is_classifier,
                 optimizer, initializer=tf.random_normal_initializer(stddev=0.1)):
        # Initialize hyperparameters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.batchnorm = batchnorm
        self.num_sampled = 20
        self.is_classifier = is_classifier
        self.optimizer = optimizer

        # Define placeholders
        self.input_x = tf.keras.Input(shape=(self.sequence_length,), dtype=tf.int32, name="input_x")
        if is_classifier:
            self.input_y = tf.keras.Input(shape=(), dtype=tf.int32, name="input_y")
        else:
            self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = self.epoch_step.assign_add(1)
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()

    def instantiate_weights(self):
        # Instantiate weights here
        pass

    def inference(self):
        # Define the inference model here
        pass

    def loss(self):
        # Define the loss function here
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])
        return tf.reduce_mean(tf.nn.nce_loss(weights=self.weight,
                                            biases=self.biases,
                                            labels=self.input_y,
                                            inputs=self.logits,
                                            num_sampled=self.num_sampled,
                                            num_classes=self.num_classes)) + self.l2_reg * l2_losses

    def train_op(self):
        # Define the training operation here
        pass

    def test_op(self):
        # Define the test operation here
        pass