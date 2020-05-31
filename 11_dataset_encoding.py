from skimage.transform import resize
import numpy as np
from skimage import io
import random, glob, os, pickle
from keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from files.utilities import printProgressBar

random_seed = 1
batch_size = 16
num_classes = 2
nImageRows = 100
nImageCols = 100
X_train = []
Y_train = []
X_test = []
Y_test = []

np.random.seed(random_seed)

positiveTrainSamples = glob.glob('./dataset/7_main/train/drunk/*')
print("Drunk Train Samples :",len(positiveTrainSamples))
negativeTrainSamples = glob.glob('./dataset/7_main/train/sober/*')
print("Sober Train Samples :",len(negativeTrainSamples))
positiveTestSamples = glob.glob('./dataset/7_main/test/drunk/*')
print("Drunk Test Samples  :",len(positiveTestSamples))
negativeTestSamples = glob.glob('./dataset/7_main/test/sober/*')
print("Sober Test Samples  :",len(negativeTestSamples))

#UNCOMMENT TO ADJUST SAMPLE SIZE TO BE EQUAL
"""
adjusted_train = len(positiveTrainSamples)
if adjusted_train > len(negativeTrainSamples):
    adjusted_train = len(negativeTrainSamples)
adjusted_train = round(int(adjusted_train * 10 ** -1) / 10 ** -1)
adjusted_test = len(positiveTestSamples)
if adjusted_test > len(negativeTestSamples):
    adjusted_test = negativeTestSamples
adjusted_test = round(int(adjusted_test * 10 ** -1) / 10 ** -1)

while len(positiveTrainSamples)>adjusted_train:
    rand = random.randint(0,len(positiveTrainSamples)-1)
    del positiveTrainSamples[rand]
print("Adjusted Drunk Train Samples :",len(positiveTrainSamples))
while len(negativeTrainSamples)>adjusted_train:
    rand = random.randint(0,len(negativeTrainSamples)-1)
    del negativeTrainSamples[rand]
print("Adjusted Sober Train Samples :",len(negativeTrainSamples))

while len(positiveTestSamples)>adjusted_test:
    rand = random.randint(0,len(positiveTestSamples)-1)
    del positiveTestSamples[rand]
print("Adjusted Drunk Test Samples  :",len(positiveTestSamples))
while len(negativeTestSamples)>adjusted_test:
    rand = random.randint(0,len(negativeTestSamples)-1)
    del negativeTestSamples[rand]
print("Adjusted Sober Test Samples  :",len(negativeTestSamples))
"""

count = 1
printProgressBar(0, len(positiveTrainSamples), prefix = 'Reading Drunk Train Samples:', suffix = 'Complete', length = 50)
for i in range(len(positiveTrainSamples)):
    image = io.imread(positiveTrainSamples[i])
    if len(image.shape) == 3 and image.shape[2]==3:
        X_train.append(resize(image, (nImageRows, nImageCols)))
        Y_train.append(1)
        printProgressBar(count, len(positiveTrainSamples), prefix = 'Reading Drunk Train Samples  :', suffix = 'Complete', length = 50)
        count = count+1

count = 1
printProgressBar(0, len(negativeTrainSamples), prefix = 'Reading Sober Train Samples:', suffix = 'Complete', length = 50)
for i in range(len(negativeTrainSamples)):
    image = io.imread(negativeTrainSamples[i])
    if len(image.shape) == 3 and image.shape[2]==3:
        X_train.append(resize(image, (nImageRows, nImageCols)))
        Y_train.append(0)
        printProgressBar(count, len(negativeTrainSamples), prefix = 'Reading Sober Train Samples  :', suffix = 'Complete', length = 50)
        count = count+1

X_train = np.array(X_train)
Y_train = np.array(Y_train)

count = 1
printProgressBar(0, len(positiveTestSamples), prefix = 'Reading Drunk Test Samples:', suffix = 'Complete', length = 50)
for i in range(len(positiveTestSamples)):
    image = io.imread(positiveTestSamples[i])
    if len(image.shape) == 3 and image.shape[2]==3:
        X_test.append(resize(image, (nImageRows, nImageCols)))
        Y_test.append(1)
        printProgressBar(count, len(positiveTestSamples), prefix = 'Reading Drunk Test Samples   :', suffix = 'Complete', length = 50)
        count = count+1

count = 1
printProgressBar(0, len(negativeTestSamples), prefix = 'Reading Sober Test Samples:', suffix = 'Complete', length = 50)
for i in range(len(negativeTestSamples)):
    image = io.imread(negativeTestSamples[i])
    if len(image.shape) == 3 and image.shape[2]==3:
        X_test.append(resize(image, (nImageRows, nImageCols)))
        Y_test.append(0)
        printProgressBar(count, len(negativeTestSamples), prefix = 'Reading Sober Test Samples   :', suffix = 'Complete', length = 50)
        count = count+1

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

print("Saving Pickle Files")
f = open("./files/x_train.pickle", "wb")
f.write(pickle.dumps(X_train))
f.close()
print("Saved ./files/x_train.pickle")
f = open("./files/y_train.pickle", "wb")
f.write(pickle.dumps(Y_train))
f.close()
print("Saved ./files/y_train.pickle")
f = open("./files/x_test.pickle", "wb")
f.write(pickle.dumps(X_test))
f.close()
print("Saved ./files/x_test.pickle")
f = open("./files/y_test.pickle", "wb")
f.write(pickle.dumps(Y_test))
f.close()
print("Saved ./files/y_test.pickle")
