# *-* coding: utf-8 *-

import tensorflow as tf


class LeNet5:

    def __init__(self):

        self.learning_rate = 0.001
        self.channels = 1
        self.random_mean = 0
        self.random_stddev = 0.1

    def create_network(self):

        # Input.

        X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, self.channels], name='X')

        # Labels.

        Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')

        # Layer 1.

        conv1_filters = tf.Variable(self.__random_values(shape=[5, 5, self.channels, 6]), name='conv1_filters')
        conv1_convs = tf.nn.conv2d(input=X, filter=conv1_filters, strides=[1, 1, 1, 1], padding='VALID',
                                   name='conv1_convs')
        conv1_biases = tf.Variable(tf.zeros(shape=[6], dtype=tf.float32), name='conv1_biases')
        conv1_preactivations = tf.add(conv1_convs, conv1_biases, name='conv1_preactivations')
        conv1_activations = tf.nn.relu(conv1_preactivations, name='conv1_activations')

        pool1 = tf.nn.max_pool(conv1_activations, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='pool1')

        # Layer 2.

        conv2_filters = tf.Variable(self.__random_values(shape=[5, 5, 6, 16]), name='conv2_filters')
        conv2_convs = tf.nn.conv2d(input=pool1, filter=conv2_filters, strides=[1, 1, 1, 1], padding='VALID',
                                   name='conv2_convs')
        conv2_biases = tf.Variable(initial_value=tf.zeros(shape=[16], dtype=tf.float32), name='conv2_biases')
        conv2_preactivations = tf.add(conv2_convs, conv2_biases, name='conv2_preactivations')
        conv2_activations = tf.nn.relu(conv2_preactivations, name='conv2_activations')

        pool2 = tf.nn.max_pool(conv2_activations, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='pool2')

        # Layer 3.

        fc3_wights = tf.Variable(self.__random_values(shape=[400, 120]), name='fc3_wights')
        fc3_biases = tf.Variable(tf.zeros(shape=[120], dtype=tf.float32), name='fc3_biases')
        fc3_preactivations = tf.add(tf.matmul(tf.reshape(pool2, shape=[-1, 400]), fc3_wights), fc3_biases,
                                    name='fc3_preactivations')
        fc3_activations = tf.nn.relu(fc3_preactivations, name='fc3_activations')

        # Layer 4.

        fc4_wights = tf.Variable(self.__random_values(shape=[120, 84]), name='fc4_wights')
        fc4_biases = tf.Variable(tf.zeros(shape=[84], dtype=tf.float32), name='fc4_biases')
        fc4_preactivations = tf.add(tf.matmul(fc3_activations, fc4_wights), fc4_biases, name='fc4_preactivations')
        fc4_activations = tf.nn.relu(fc4_preactivations, name='fc4_activations')

        # Layer 5.

        fc5_wights = tf.Variable(self.__random_values(shape=[84, 10]), name='fc5_wights')
        fc5_biases = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32), name='fc5_biases')
        fc5_logits = tf.add(tf.matmul(fc4_activations, fc5_wights), fc5_biases, name='fc5_logits')

        # Training.

        logits = fc5_logits
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='cross_entropy')
        loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_operation = optimizer.minimize(loss_operation, name='training_operation')

        # Accuracy.

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1), name='correct_prediction')
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')

        # Network.

        return X, Y, logits, training_operation, accuracy_operation

    def train_epoch(self, sess, X, Y, training_operation, X_data, Y_data, batch_size):
        num_examples = len(X_data)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            sess.run(training_operation, feed_dict={X: batch_x, Y: batch_y})
        return

    def evaluate(self, sess, X, Y, accuracy_operation, X_data, Y_data, batch_size):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            accuracy = sess.run(accuracy_operation, feed_dict={X: batch_x, Y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def __random_values(self, shape):
        return tf.random_normal(shape=shape, mean=self.random_mean, stddev=self.random_stddev, dtype=tf.float32)
