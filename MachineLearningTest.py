import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

distribution_train = []
probs_train = []
# x_train = []
# y_train = []

with open('tftest.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        distribution_train.append(row[0])
        probs_train.append(row[1])

distribution_train.pop(0)
probs_train.pop(0)

# Flattens and converts string in matrices into numbers
def num_converter_flatten(csv_list):
    stockfish = []
    for j in range(len(csv_list)):
        append_this = []
        for i in csv_list[j]:
            if i == '1' or i == '2' or i == '3' or i == '4' or i == '5' or i == '6' or i == '7' or i == '8' or i =='9' or i =='0':
                append_this.append(float(i))
        stockfish.append((append_this))

    return stockfish

x_train = num_converter_flatten(distribution_train)
y_train = num_converter_flatten(probs_train)

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# y_train = tf.keras.utils.normalize(y_train, axis=1)
#
# model = tf.keras.models.Sequential()
#
# model.add(tf.keras.layers.Flatten())
#
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#
# model.add(tf.keras.layers.Dense(80, activation=tf.nn.softmax))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
#
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss)
# print(val_acc)