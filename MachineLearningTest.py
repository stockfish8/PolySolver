'''Imports'''
import csv
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
# np.set_printoptions(suppress=True)
import keras.backend as K
from keras.metrics import binary_accuracy
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

''' Here we unpack the .csv files. I have chosen to put their contents into lists.
 Do let me know if there exists a more efficient method. '''
distribution_train = []
probs_train = []
distribution_test = []
probs_test = []

with open('training_sample.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        distribution_train.append(row[0])
        probs_train.append(row[1])

with open('testing_sample.csv') as csv_file_1:
    csv_reader_1 = csv.reader(csv_file_1, delimiter= ',')

    for row in csv_reader_1:
        distribution_test.append(row[0])
        probs_test.append(row[1])


'''Get rid of the titles in the training_sample.csv file.'''
distribution_train.pop(0)
probs_train.pop(0)

'''For some reason everything in my csv file is stored as strings. Or maybe it's just because of the
way I have unpacked it. The below function is to convert it into floats so that TF can work with it.
It's crude, but it locates all the numbers and appends them to a list, which then gets appended to
a giant list called f.'''

def num_converter_flatten(csv_list):
    f = []
    for j in range(len(csv_list)):
        append_this = []
        for i in csv_list[j]:
            if i == '1' or i == '2' or i == '3' or i == '4' or i == '5' or i == '6' or i == '7' or i == '8' or i =='9' or i =='0':
                append_this.append(float(i))
        f.append((append_this))

    return f

def custom_metric(y_true, y_pred):
    threshold =0.5
    thresholded_values = K.greater(y_pred, threshold)
    return binary_accuracy(y_true, thresholded_values)

x_train = num_converter_flatten(distribution_train)
y_train = num_converter_flatten(probs_train)

x_train = np.array(x_train)/5
y_train = np.array(y_train)

# print (x_train)
# print (y_train)

x_test = num_converter_flatten(distribution_test)
y_test = num_converter_flatten(probs_test)
x_test = np.array(x_test)/5
y_test = np.array(y_test)

'''Model starts from here'''

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(500, input_dim=26,activation='relu'))
model.add(tf.keras.layers.Dense(80, activation='sigmoid'))

model.compile(optimizer='rmsprop',
          loss='binary_crossentropy',
          metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

val_loss, val_acc = model.evaluate(x_test, y_test)
print (val_loss, val_acc)

model.save('epic_equation_model_try1')
new_model = tf.keras.models.load_model('epic_equation_model_try1')
predictions = new_model.predict(x_test)

print (np.array(predictions[1]))



