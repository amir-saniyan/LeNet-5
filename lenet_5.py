# -*- coding: utf-8 -*-

import tensorflow as tf


class LeNet5:

    def __init__(self, input_width=32, input_height=32, input_channels=1, num_classes=10, learning_rate=0.001):

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.random_mean = 0
        self.random_stddev = 0.1

        # ----------------------------------------------------------------------------------------------------

        # Input: 32x32x1.
        with tf.name_scope('input'):
            self.X = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.input_height, self.input_width, self.input_channels], name='X')

        # Labels: 10.
        with tf.name_scope('labels'):
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='Y')

        # Layer 1.
        # [Input] ==> 32x32x1
        # --> 32x32x1 ==> [Convolution: size=(5x5x1)x6, strides=1, padding=valid] ==> 28x28x6
        # --> 28x28x6 ==> [ReLU] ==> 28x28x6
        # --> 28x28x6 ==> [Max-Pool: size=2x2, strides=2, padding=valid] ==> 14x14x6
        # --> [Output] ==> 14x14x6
        with tf.name_scope('layer1'):
            layer1_activations = self.__conv(input=self.X, filter_width=5, filter_height=5, filters_count=6, stride_x=1,
                                             stride_y=1, padding='VALID')
            layer1_pool = self.__max_pool(layer1_activations, filter_width=2, filter_height=2, stride_x=2, stride_y=2,
                                          padding='VALID')

        # Layer 2.
        # [Input] ==> 14x14x6
        # --> 14x14x6 ==> [Convolution: size=(5x5x6)x16, strides=1, padding=valid] ==> 10x10x16
        # --> 10x10x16 ==> [ReLU] ==> 10x10x16
        # --> 10x10x16 ==> [Max-Pool: size=2x2, strides=2, padding=valid] ==> 5x5x16
        # --> [Output] ==> 5x5x16
        with tf.name_scope('layer2'):
            layer2_activations = self.__conv(input=layer1_pool, filter_width=5, filter_height=5, filters_count=16,
                                             stride_x=1, stride_y=1, padding='VALID')
            layer2_pool = self.__max_pool(layer2_activations, filter_width=2, filter_height=2, stride_x=2, stride_y=2,
                                          padding='VALID')

        # Layer 3.
        # [Input] ==> 5x5x16=400
        # --> 400 ==> [Fully Connected: neurons=120] ==> 120
        # --> 120 ==> [ReLU] ==> 120
        # --> [Output] ==> 120
        with tf.name_scope('layer3'):
            pool2_shape = layer2_pool.get_shape().as_list()
            flattened_input_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
            layer3_fc = self.__fully_connected(input=tf.reshape(layer2_pool, shape=[-1, flattened_input_size]),
                                               inputs_count=flattened_input_size, outputs_count=120, relu=True)

        # Layer 4.
        # [Input] ==> 120
        # --> 120 ==> [Fully Connected: neurons=84] ==> 84
        # --> 84 ==> [ReLU] ==> 84
        # --> [Output] ==> 84
        with tf.name_scope('layer4'):
            layer4_fc = self.__fully_connected(input=layer3_fc, inputs_count=120, outputs_count=84, relu=True)

        # Layer 5.
        # [Input] ==> 84
        # --> 84 ==> [Logits: neurons=10] ==> 10
        # --> [Output] ==> 10
        with tf.name_scope('layer5'):
            layer5_logits = self.__fully_connected(input=layer4_fc, inputs_count=84, outputs_count=self.num_classes,
                                                   relu=False, name='logits')

        # Cross Entropy.
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer5_logits, labels=self.Y,
                                                                       name='cross_entropy')
            self.__variable_summaries(cross_entropy)

        # Training.
        with tf.name_scope('training'):
            loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
            tf.summary.scalar(name='loss', tensor=loss_operation)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # self.training_operation = optimizer.minimize(loss_operation, name='training_operation')

            grads_and_vars = optimizer.compute_gradients(loss_operation)
            self.training_operation = optimizer.apply_gradients(grads_and_vars, name='training_operation')

            for grad, var in grads_and_vars:
                if grad is not None:
                    with tf.name_scope(var.op.name + '/gradients'):
                        self.__variable_summaries(grad)

        # Accuracy.
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(layer5_logits, 1), tf.argmax(self.Y, 1), name='correct_prediction')
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
            tf.summary.scalar(name='accuracy', tensor=self.accuracy_operation)

    def train_epoch(self, sess, X_data, Y_data, batch_size, file_writer=None, summary_operation=None,
                    epoch_number=None):
        num_examples = len(X_data)
        step = 0
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            if file_writer is not None and summary_operation is not None:
                _, summary = sess.run([self.training_operation, summary_operation],
                                      feed_dict={self.X: batch_x, self.Y: batch_y})
                file_writer.add_summary(summary, epoch_number * (num_examples // batch_size + 1) + step)
                step += 1
            else:
                sess.run(self.training_operation, feed_dict={self.X: batch_x, self.Y: batch_y})

    def evaluate(self, sess, X_data, Y_data, batch_size=128):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            batch_accuracy = sess.run(self.accuracy_operation, feed_dict={self.X: batch_x, self.Y: batch_y})
            total_accuracy += (batch_accuracy * len(batch_x))
        return total_accuracy / num_examples

    def save(self, sess, file_name):
        saver = tf.train.Saver()
        saver.save(sess, file_name)

    def restore(self, sess, checkpoint_dir):
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    def __random_values(self, shape):
        return tf.random_normal(shape=shape, mean=self.random_mean, stddev=self.random_stddev, dtype=tf.float32)

    def __variable_summaries(self, var):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)

    def __conv(self, input, filter_width, filter_height, filters_count, stride_x, stride_y, padding='VALID',
               name='conv'):
        with tf.name_scope(name):
            input_channels = input.get_shape()[-1].value
            filters = tf.Variable(
                self.__random_values(shape=[filter_height, filter_width, input_channels, filters_count]),
                name='filters')
            convs = tf.nn.conv2d(input=input, filter=filters, strides=[1, stride_y, stride_x, 1], padding=padding,
                                 name='convs')
            biases = tf.Variable(tf.zeros(shape=[filters_count], dtype=tf.float32), name='biases')
            preactivations = tf.nn.bias_add(convs, biases, name='preactivations')
            activations = tf.nn.relu(preactivations, name='activations')

            with tf.name_scope('filter_summaries'):
                self.__variable_summaries(filters)

            with tf.name_scope('bias_summaries'):
                self.__variable_summaries(biases)

            with tf.name_scope('preactivations_histogram'):
                tf.summary.histogram('preactivations', preactivations)

            with tf.name_scope('activations_histogram'):
                tf.summary.histogram('activations', activations)

            return activations

    def __max_pool(self, input, filter_width, filter_height, stride_x, stride_y, padding='VALID', name='pool'):
        with tf.name_scope(name):
            pool = tf.nn.max_pool(input, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                                  padding=padding, name='pool')
            return pool

    def __fully_connected(self, input, inputs_count, outputs_count, relu=True, name='fully_connected'):
        with tf.name_scope(name):
            wights = tf.Variable(self.__random_values(shape=[inputs_count, outputs_count]), name='wights')
            biases = tf.Variable(tf.zeros(shape=[outputs_count], dtype=tf.float32), name='biases')
            preactivations = tf.nn.bias_add(tf.matmul(input, wights), biases, name='preactivations')
            if relu:
                activations = tf.nn.relu(preactivations, name='activations')

            with tf.name_scope('wight_summaries'):
                self.__variable_summaries(wights)

            with tf.name_scope('bias_summaries'):
                self.__variable_summaries(biases)

            with tf.name_scope('preactivations_histogram'):
                tf.summary.histogram('preactivations', preactivations)

            if relu:
                with tf.name_scope('activations_histogram'):
                    tf.summary.histogram('activations', activations)

            if relu:
                return activations
            else:
                return preactivations
