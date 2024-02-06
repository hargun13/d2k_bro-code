# # import tensorflow as tf
# #
# # from loss import errors_mean
# #
# # class TextCNN(object):
# #     def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, batchnorm, decay_steps,
# #                  decay_rate, sequence_length, vocab_size, embed_size, is_training, is_classifier, optimizer,
# #                  clip_gradients=5.0, decay_rate_big=0.50, initializer=tf.truncated_normal_initializer(stddev=0.1)):
# #
# #         # set hyperparameters
# #         self.num_classes = num_classes
# #         self.batch_size = batch_size
# #         self.sequence_length = sequence_length
# #         self.vocab_size = vocab_size
# #         self.embed_size = embed_size
# #         self.is_training = is_training
# #         self.is_classifier = is_classifier
# #         self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
# #         self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
# #
# #         self.filter_sizes = filter_sizes
# #         self.num_filters = num_filters
# #         self.initializer = initializer
# #         self.num_filters_total = self.num_filters * len(filter_sizes)
# #         self.clip_gradients = clip_gradients
# #
# #         self.optimizer = optimizer
# #         self.batchnorm = batchnorm
# #         if optimizer == 'Adam':
# #             self.learning_rate = 0.001
# #
# #         self.instantiate_weights()
# #         self.logits = self.inference()
# #         if not is_training:
# #             return
# #         self.loss_val = self.loss()
# #         self.train_op = self.train()
# #         if self.is_classifier:
# #             self.predictions = tf.argmax(self.logits, 1, name="predictions")
# #             correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
# #             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
# #         else:
# #             self.accuracy = tf.constant(0.5)
# #
# #     def loss(self, l2_lambda=0.0001):
# #         with tf.name_scope("loss"):
# #             if self.is_classifier:
# #                 losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
# #                 loss = tf.reduce_mean(losses)
# #                 l2_losses = tf.add_n(
# #                     [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
# #                 loss = loss + l2_losses
# #             else:
# #                 loss = errors_mean(self.input_y, self.logits)
# #         return loss
# #
# #     def instantiate_weights(self):
# #         with tf.name_scope("embedding"):
# #             self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
# #                                              initializer=self.initializer)
# #             self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
# #                                                 initializer=self.initializer)
# #             self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])
# #
# #     def train(self):
# #         learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
# #                                                    self.decay_rate, staircase=True)
# #         train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
# #                                                    learning_rate=learning_rate,
# #                                                    optimizer=self.optimizer, clip_gradients=self.clip_gradients)
# #         return train_op
# #
# #     def batch_norm(self, x, n_out, phase_train):
# #         with tf.variable_scope('bn'):
# #             beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
# #             gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
# #             batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
# #             ema = tf.train.ExponentialMovingAverage(decay=0.5)
# #             def mean_var_with_update():
# #                 ema_apply_op = ema.apply([batch_mean, batch_var])
# #                 with tf.control_dependencies([ema_apply_op]):
# #                     return tf.identity(batch_mean), tf.identity(batch_var)
# #             mean, var = tf.cond(phase_train, mean_var_with_update,
# #                                 lambda: (ema.average(batch_mean), ema.average(batch_var)))
# #             bnormed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
# #         return bnormed
# #
# #     def inference(self):
# #         self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
# #         self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)
# #
# #         pooled_outputs = []
# #         for i, filter_size in enumerate(self.filter_sizes):
# #             with tf.name_scope("convolution-pooling-%s" % filter_size):
# #                 filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1,
# #                                                                      self.num_filters],
# #                                          initializer=self.initializer)
# #                 conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],
# #                                     padding="VALID", name="conv")
# #                 b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
# #
# #                 if self.batchnorm:
# #                     conv_bn = self.batch_norm(conv, self.num_filters, tf.cast(self.is_training, tf.bool))
# #                     h = tf.nn.relu(tf.nn.bias_add(conv_bn, b), "relu")
# #                 else:
# #                     h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
# #
# #                 pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
# #                                         strides=[1, 1, 1, 1], padding='VALID', name="pool")
# #                 pooled_outputs.append(pooled)
# #
# #         self.h_pool = tf.concat(pooled_outputs, 3)
# #         self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
# #
# #         with tf.name_scope("dropout"):
# #             self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)
# #
# #         with tf.name_scope("output"):
# #             logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
# #         return logits
# #
# # # # Placeholder for loss.py import
# # # # Placeholder for data preprocessing and handling
# # #
# # # # Main code entry point
# # # if __name__ == "__main__":
# # #     # Placeholder for data preprocessing and handling
# # #     # Placeholder for model initialization, training, and evaluation
# #
# import tensorflow as tf
# from loss import errors_mean
#
# class TextCNN(object):
#     def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, batchnorm, decay_steps,
#                  decay_rate, sequence_length, vocab_size, embed_size, is_training, is_classifier, optimizer,
#                  clip_gradients=5.0, decay_rate_big=0.50, initializer=tf.random_normal_initializer(stddev=0.1)):
#
#         # set hyperparameters
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.sequence_length = sequence_length
#         self.vocab_size = vocab_size
#         self.embed_size = embed_size
#         self.is_training = is_training
#         self.is_classifier = is_classifier
#         self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
#         self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
#
#         self.filter_sizes = filter_sizes
#         self.num_filters = num_filters
#         self.initializer = initializer
#         self.num_filters_total = self.num_filters * len(filter_sizes)
#         self.clip_gradients = clip_gradients
#
#         self.optimizer = optimizer
#         self.batchnorm = batchnorm
#         if optimizer == 'Adam':
#             self.learning_rate = 0.001
#

#         self.instantiate_weights()
#         self.logits = self.inference()
#         if not is_training:
#             return
#         self.loss_val = self.loss()
#         self.train_op = self.train()
#         if self.is_classifier:
#             self.predictions = tf.argmax(self.logits, 1, name="predictions")
#             correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
#             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
#         else:
#             self.accuracy = tf.constant(0.5)
#
#     def loss(self, l2_lambda=0.0001):
#         with tf.name_scope("loss"):
#             if self.is_classifier:
#                 losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
#                 loss = tf.reduce_mean(losses)
#                 l2_losses = tf.add_n(
#                     [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
#                 loss = loss + l2_losses
#             else:
#                 loss = errors_mean(self.input_y, self.logits)
#         return loss
#
#     def instantiate_weights(self):
#         with tf.name_scope("embedding"):
#             self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
#                                              initializer=self.initializer)
#             self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
#                                                 initializer=self.initializer)
#             self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])
#
#     def train(self):
#         learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
#                                                    self.decay_rate, staircase=True)
#         train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
#                                                    learning_rate=learning_rate,
#                                                    optimizer=self.optimizer, clip_gradients=self.clip_gradients)
#         return train_op
#
#     def batch_norm(self, x, n_out, phase_train):
#         with tf.variable_scope('bn'):
#             beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
#             gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
#             batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
#             ema = tf.train.ExponentialMovingAverage(decay=0.5)
#             def mean_var_with_update():
#                 ema_apply_op = ema.apply([batch_mean, batch_var])
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)
#             mean, var = tf.cond(phase_train, mean_var_with_update,
#                                 lambda: (ema.average(batch_mean), ema.average(batch_var)))
#             bnormed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#         return bnormed
#
#     def inference(self):
#         self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
#         self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)
#
#         pooled_outputs = []
#         for i, filter_size in enumerate(self.filter_sizes):
#             with tf.name_scope("convolution-pooling-%s" % filter_size):
#                 filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1,
#                                                                      self.num_filters],
#                                          initializer=self.initializer)
#                 conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],
#                                     padding="VALID", name="conv")
#                 b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
#
#                 if self.batchnorm:
#                     conv_bn = self.batch_norm(conv, self.num_filters, tf.cast(self.is_training, tf.bool))
#                     h = tf.nn.relu(tf.nn.bias_add(conv_bn, b), "relu")
#                 else:
#                     h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
#
#                 pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
#                                         strides=[1, 1, 1, 1], padding='VALID', name="pool")
#                 pooled_outputs.append(pooled)
#
#         self.h_pool = tf.concat(pooled_outputs, 3)
#         self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
#
#         with tf.name_scope("dropout"):
#             self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)
#
#         with tf.name_scope("output"):
#             logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
#         return logits



import tensorflow as tf
from loss import errors_mean
from tensorflow.compat.v1.assign_ops import assign
class TextCNN(object):
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, batchnorm, decay_steps,
                 decay_rate, sequence_length, vocab_size, embed_size, is_training, is_classifier, optimizer,
                 clip_gradients=5.0, decay_rate_big=0.50, initializer=tf.random_normal_initializer(stddev=0.1)):

        # set hyperparameters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        # self.learning_rate_decay_half_op = tf.compat.v1.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        # self.learning_rate_decay_half_op = assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.learning_rate_decay_half_op = self.learning_rate.assign(self.learning_rate * decay_rate_big)

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.clip_gradients = clip_gradients

        self.optimizer = optimizer
        self.batchnorm = batchnorm
        if optimizer == 'Adam':
            self.learning_rate = 0.001

        self.instantiate_weights()
        self.logits = self.inference()
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        if self.is_classifier:
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            self.accuracy = tf.constant(0.5)

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            if self.is_classifier:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
                loss = tf.reduce_mean(losses)
                l2_losses = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
                loss = loss + l2_losses
            else:
                loss = errors_mean(self.input_y, self.logits)
        return loss

    def instantiate_weights(self):
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate,
                                                   optimizer=self.optimizer, clip_gradients=self.clip_gradients)
        return train_op

    def batch_norm(self, x, n_out, phase_train):
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            bnormed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return bnormed

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1,
                                                                     self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])

                if self.batchnorm:
                    conv_bn = self.batch_norm(conv, self.num_filters, tf.cast(self.is_training, tf.bool))
                    h = tf.nn.relu(tf.nn.bias_add(conv_bn, b), "relu")
                else:
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits

