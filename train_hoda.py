# -*- coding: utf-8 -*-

import tensorflow as tf

from lenet_5 import LeNet5
from hoda_dataset_helper import read_hoda

INPUT_WIDTH = 32
INPUT_HEIGHT = 32
INPUT_CHANNELS = 1

NUM_CLASSES = 10

LEARNING_RATE = 0.001

EPOCHS = 100
BATCH_SIZE = 128

X_train, Y_train, X_test, Y_test, X_remaining, Y_remaining = read_hoda()

lenet5 = LeNet5(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, input_channels=INPUT_CHANNELS,
                num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

with tf.Session() as sess:
    print('Training hoda dataset...')
    print()

    file_writer = tf.summary.FileWriter(logdir='./hoda/log', graph=sess.graph)

    summary_operation = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):

        print('Calculating accuracies...')

        train_accuracy = lenet5.evaluate(sess, X_train, Y_train, BATCH_SIZE)
        test_accuracy = lenet5.evaluate(sess, X_test, Y_test, BATCH_SIZE)
        remaining_accuracy = lenet5.evaluate(sess, X_remaining, Y_remaining, BATCH_SIZE)

        print('Train Accuracy = {:.3f}'.format(train_accuracy))
        print('Test Accuracy = {:.3f}'.format(test_accuracy))
        print('Remaining Accuracy = {:.3f}'.format(remaining_accuracy))
        print()

        print('Training epoch', i + 1, '...')
        lenet5.train_epoch(sess, X_train, Y_train, BATCH_SIZE, file_writer, summary_operation, i)
        print()

    final_train_accuracy = lenet5.evaluate(sess, X_train, Y_train, BATCH_SIZE)
    final_test_accuracy = lenet5.evaluate(sess, X_test, Y_test, BATCH_SIZE)
    final_remaining_accuracy = lenet5.evaluate(sess, X_remaining, Y_remaining, BATCH_SIZE)

    print('Final Train Accuracy = {:.3f}'.format(final_train_accuracy))
    print('Final Test Accuracy = {:.3f}'.format(final_test_accuracy))
    print('Final Remaining Accuracy = {:.3f}'.format(final_remaining_accuracy))
    print()

    lenet5.save(sess, './hoda/model/lenet5')
    print('Model saved.')
    print()

print('Training done successfully.')
