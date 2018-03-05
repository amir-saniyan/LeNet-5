# *-* coding: utf-8 *-

import tensorflow as tf

from lenet_5 import LeNet5
from mnist_dataset_helper import read_mnist


EPOCHS = 100
BATCH_SIZE = 128

lenet5 = LeNet5()
X, Y, logits, training_operation, accuracy_operation = lenet5.create_network()

X_train, Y_train, X_validation, Y_validation, X_test, Y_test = read_mnist()

with tf.Session() as sess:
    print('Training...')
    print()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):

        print('Calculating accuracies...')

        train_accuracy = lenet5.evaluate(sess, X, Y, accuracy_operation, X_train, Y_train, BATCH_SIZE)
        validation_accuracy = lenet5.evaluate(sess, X, Y, accuracy_operation, X_validation, Y_validation, BATCH_SIZE)
        test_accuracy = lenet5.evaluate(sess, X, Y, accuracy_operation, X_test, Y_test, BATCH_SIZE)

        print('Train Accuracy = {:.3f}'.format(train_accuracy))
        print('Validation Accuracy = {:.3f}'.format(validation_accuracy))
        print('Test Accuracy = {:.3f}'.format(test_accuracy))
        print()

        print('Training epoch', i + 1, '...')
        print()
        lenet5.train_epoch(sess, X, Y, training_operation, X_train, Y_train, BATCH_SIZE)

    final_train_accuracy = lenet5.evaluate(sess, X, Y, accuracy_operation, X_train, Y_train, BATCH_SIZE)
    final_validation_accuracy = lenet5.evaluate(sess, X, Y, accuracy_operation, X_validation, Y_validation, BATCH_SIZE)
    final_test_accuracy = lenet5.evaluate(sess, X, Y, accuracy_operation, X_test, Y_test, BATCH_SIZE)

    print('Final Train Accuracy = {:.3f}'.format(final_train_accuracy))
    print('Final Validation Accuracy = {:.3f}'.format(final_validation_accuracy))
    print('Final Test Accuracy = {:.3f}'.format(final_test_accuracy))
    print()

    saver.save(sess, './lenet5')
    print('Model saved.')

print('OK')
