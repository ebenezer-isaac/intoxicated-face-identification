from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import optimizers
from skimage import io
from skimage.transform import resize
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import random, glob, os
from utilities import printProgressBar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
tf.test.gpu_device_name()

epochs = 30
nChannels = 3
random_seed = 1
batch_size = 2
num_classes = 2
nImageRows = 100
nImageCols = 100
log_filepath = 'log.txt'
X_train = []
Y_train = []
X_test = []
Y_test = []

tf.random.set_seed(random_seed)
np.random.seed(random_seed)

positiveTrainSamples = glob.glob('../data/4_split/train/drunk/*')
print("Drunk Train Samples :",len(positiveTrainSamples))
count = 1
printProgressBar(0, len(positiveTrainSamples), prefix = 'Reading Drunk Train Samples:', suffix = 'Complete', length = 50)
for i in range(len(positiveTrainSamples)):
    X_train.append(resize(io.imread(positiveTrainSamples[i]), (nImageRows, nImageCols)))
    Y_train.append(1)
    printProgressBar(count, len(positiveTrainSamples), prefix = 'Reading Drunk Train Samples:', suffix = 'Complete', length = 50)
    count = count+1

negativeTrainSamples = glob.glob('../data/4_split/train/sober/*')
print("Sober Train Samples :",len(negativeTrainSamples))
count = 1
printProgressBar(0, len(negativeTrainSamples), prefix = 'Reading Sober Train Samples:', suffix = 'Complete', length = 50)
for i in range(len(negativeTrainSamples)):
    X_train.append(resize(io.imread(negativeTrainSamples[i]), (nImageRows, nImageCols)))
    Y_train.append(0)
    printProgressBar(count, len(negativeTrainSamples), prefix = 'Reading Sober Train Samples:', suffix = 'Complete', length = 50)
    count = count+1

positiveTestSamples = glob.glob('../data/4_split/test/drunk/*')
print("Drunk Test Samples :",len(positiveTestSamples))
count = 1
printProgressBar(0, len(positiveTestSamples), prefix = 'Reading Drunk Test Samples:', suffix = 'Complete', length = 50)
for i in range(len(positiveTestSamples)):
    X_test.append(resize(io.imread(positiveTestSamples[i]), (nImageRows, nImageCols)))
    Y_test.append(1)
    printProgressBar(count, len(positiveTestSamples), prefix = 'Reading Drunk Test Samples:', suffix = 'Complete', length = 50)
    count = count+1

negativeTestSamples = glob.glob('../data/4_split/test/sober/*')
print("Sober Test Samples :",len(negativeTestSamples))
count = 1
printProgressBar(0, len(negativeTestSamples), prefix = 'Reading Sober Test Samples:', suffix = 'Complete', length = 50)
for i in range(len(negativeTestSamples)):
    X_test.append(resize(io.imread(negativeTestSamples[i]), (nImageRows, nImageCols)))
    Y_test.append(0)
    printProgressBar(count, len(negativeTestSamples), prefix = 'Reading Sober Test Samples:', suffix = 'Complete', length = 50)
    count = count+1

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)


mean = np.array([0.5,0.5,0.5])
std = np.array([1,1,1])
X_train = X_train.astype('float')
X_test = X_test.astype('float')
for i in range(3):
    X_train[:,:,:,i] = (X_train[:,:,:,i]- mean[i]) / std[i]
    X_test[:,:,:,i] = (X_test[:,:,:,i]- mean[i]) / std[i]
num_iterations = int(len(X_train)/batch_size) + 1
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)


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
model.save_weights('model4.h5')

print("Model Fit :",model_fit)

print("Score :", score)
print("Test Loss :", score[0])
print("Test Accuracy :", score[1])