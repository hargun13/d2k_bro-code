import tensorflow as tf

from loss import errors_mean

class FastText:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 num_sampled, sequence_length, vocab_size, embed_size, is_training,
                 batchnorm, is_classifier, optimizer):
        # init all hyperparameters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.batchnorm = batchnorm
        self.is_classifier = is_classifier
        self.optimizer = optimizer

        # add placeholder (X,label)
        self.input_x = tf.keras.Input(shape=(self.sentence_len,), dtype=tf.int32, name="input_x")
        if is_classifier:
            self.input_y = tf.keras.Input(shape=(), dtype=tf.int32, name="input_y")
        else:
            self.input_y = tf.keras.Input(shape=(self.num_classes,), dtype=tf.float32, name="input_y")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.instantiate_weights()
        self.logits = self.inference()
        # [None, self.label_size]
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        # shape:[None,]
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        if is_classifier:
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        else:
            # self.accuracy = tf.constant(0.5)
            self.accuracy = tf.metrics.mean_squared_error(self.input_y, correct_prediction)

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = tf.Variable(tf.random.normal([self.vocab_size, self.embed_size]))

        self.W = tf.Variable(tf.random.normal([self.embed_size, self.num_classes]))
        self.b = tf.Variable(tf.random.normal([self.num_classes]))

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get embedding of words in the sentence
        # [None,self.sentence_len,self.embed_size]
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # 2.average vectors, to get representation of the sentence
        # [None,self.embed_size]
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)

        # 3.linear classifier layer
        # [None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b
        return logits

    def loss(self, l2_lambda=0.01):  # 0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        if self.is_classifier:
            if self.is_training:  # training
                labels = tf.reshape(self.input_y, [-1])
                labels = tf.expand_dims(labels, 1)

                loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=tf.transpose(self.W),
                                   biases=self.b,
                                   labels=labels,
                                   inputs=self.sentence_embeddings,
                                   num_sampled=self.num_sampled,
                                   num_classes=self.num_classes, partition_strategy="div"))
            else:  # eval/inference
                labels_one_hot = tf.one_hot(self.input_y, self.num_classes)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
                loss = tf.reduce_sum(loss, axis=1)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                  if 'bias' not in v.name]) * l2_lambda
        else:
            loss = errors_mean(self.input_y, self.logits)
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.global_step,
                                                                       self.decay_steps, self.decay_rate, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_op = optimizer.minimize(self.loss_val, global_step=self.global_step)
        return train_op
