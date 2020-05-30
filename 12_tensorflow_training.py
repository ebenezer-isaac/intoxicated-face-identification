from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import optimizers
import random, os, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

epochs = 10
nChannels = 3
random_seed = 1
batch_size = 16
nImageRows = 100
nImageCols = 100
log_filepath = 'log.txt'

tf.random.set_seed(random_seed)

print("Reading Files")
X_train = pickle.loads(open("./files/x_train.pickle", "rb").read())
print("X_train Length :",len(X_train))
Y_train = pickle.loads(open("./files/y_train.pickle", "rb").read())
print("Y_train Length :",len(Y_train))
X_test = pickle.loads(open("./files/x_test.pickle", "rb").read())
print("X_test Length :",len(X_test))
Y_test = pickle.loads(open("./files/y_test.pickle", "rb").read())
print("Y_test Length :",len(Y_test))

modelInputShape = (nImageRows, nImageCols, nChannels)
model = Sequential()
model.add(Conv2D(8,kernel_size=(3,3), activation='relu', strides=(1,1), padding='same', input_shape=modelInputShape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Dropout(0.25))
model.add(Conv2D(16,kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Dropout(0.25))
model.add(Conv2D(16,kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Conv2D(8,kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

sgd = optimizers.SGD(lr=.001, momentum=0.9, decay=0.000005, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
tensorBoardCallback = TensorBoard(log_dir=log_filepath, histogram_freq=0)
callbacks = [tensorBoardCallback]
model_fit = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test))
score=model.evaluate(X_test, Y_test, verbose=0)
model.save_weights('./files/model.h5')

print("Model Fit :",model_fit)
print("Score :", score)
print("Test Loss :", score[0])
print("Test Accuracy :", score[1])
