'''Imports'''
import csv
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
np.set_printoptions(suppress=True)

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

'''Basically, this line is to convert the distribution_train and probs_train which are currently strings
into numbers. And we normalize the training data.'''
x_train = num_converter_flatten(distribution_train)
y_train = num_converter_flatten(probs_train)
x_train = tf.keras.utils.normalize(x_train, axis=1)

# print (x_train)
# print (y_train)

'''This line we reshape x_train and y_train into tensors. The convertion to float 32 is also necessary as
I realised that A and B are different floats for some reason.'''
A = tf.reshape(x_train, [-1,1*26])
B = tf.reshape(y_train, [-1,1*80])
A = tf.dtypes.cast(A, dtype = tf.float32)
B = tf.dtypes.cast(B, dtype = tf.float32)

'''Doing the same thing to x_test and y_test'''

x_test = num_converter_flatten(distribution_test)
y_test = num_converter_flatten(probs_test)
C = tf.reshape(x_test, [-1,1*26])
D = tf.reshape(y_test, [-1,1*80])
C = tf.dtypes.cast(C, dtype = tf.float32)
D = tf.dtypes.cast(D, dtype = tf.float32)


'''Model starts from here'''

model = tf.keras.models.Sequential()

'''I'm not too sure if relu is the right activation function to use here. I've tried different activation
functions, but all run into the same problem described below.'''

model.add(tf.keras.layers.Dense(180, activation=keras.activations.relu, input_shape=(26,)))

model.add(tf.keras.layers.Dense(2080, activation=keras.activations.relu))

model.add(tf.keras.layers.Dense(180, activation=keras.activations.relu))

'''I'm making the final layer 80 because I want TF to output the size of the
'probs' list in the csv file'''

model.add(tf.keras.layers.Dense(80, activation=keras.activations.softplus))

'''Again I'm not sure if softplus is the best to use here. I've also tested a number
of activation functions for the last layer, and it also runs to the same problem.'''

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(A,B, epochs=2, steps_per_epoch=16)

val_loss, val_acc = model.evaluate(C,D, steps = 128)
print (val_loss, val_acc)


'''Just saving the model'''
model.save('epic_equation_model_try1')
new_model = tf.keras.models.load_model('epic_equation_model_try1')
predictions = new_model.predict(C, steps = 1)

'''This tests for only the first prediction. If you wwant to see more predictions,
change the range.'''
for i in range(1):
    MUTI = 500000
    x = np.array(predictions[i]).reshape(5,16)
    # print (x)
    PX = MUTI*x
    PX = np.round(PX, 2)
    PX[PX<0.1] = 0
    PX[PX>0.1] = 1
    PX[PX==0.1] = 1
    print (PX)


