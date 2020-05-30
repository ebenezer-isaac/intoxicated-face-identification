from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import tensorflow as tf
import glob, cv2, dlib, pickle
from imutils.face_utils import FaceAligner
from mtcnn.mtcnn import MTCNN
from files.utilities import calcBoxArea

nImageRows = 100
nImageCols = 100
nChannels = 3

modelInputShape = (nImageRows, nImageCols, nChannels)
model = Sequential()
model.add(Conv2D(8,kernel_size=(3,3), activation="relu", strides=(1,1), padding="same", input_shape=modelInputShape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
model.add(Dropout(0.25))
model.add(Conv2D(16,kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
model.add(Dropout(0.25))
model.add(Conv2D(16,kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
model.add(Conv2D(8,kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
model.add(Flatten())
model.add(Dense(10, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.load_weights("./files/model.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./files/shape.dat")
fa = FaceAligner(predictor, desiredFaceWidth=100)
mtcnn = MTCNN()
actual = []
predicted = []

states = ["sober","drunk"]
binary_dict = {0: "sober", 1: "drunk"}
for index, state in enumerate(states):
	for image_path in glob.glob("./dataset/7_main/test/"+str(state)+"/*"):
		image = cv2.imread(image_path)
		result = mtcnn.detect_faces(image)
		if result:
			result.sort(key = calcBoxArea, reverse = True)
			(x, y, w, h) = (result[0]["box"][0],result[0]["box"][1],result[0]["box"][2],result[0]["box"][3])
			faceAligned = fa.align(image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dlib.rectangle(x,y,w+x,h+y))
			face = np.expand_dims(faceAligned, axis=0)
			prediction = model.predict(face)
			maxindex = int(np.argmax(prediction))
			actual.append(state)
			predicted.append(binary_dict[maxindex])
			print("Image : "+str(image_path)+"\nPrediction : "+str(prediction)+"\nmaxindex : "+str(maxindex)+"\nExpected : "+str(state)+"\nPredicted : "+str(binary_dict[maxindex]))
		else:
			print("Image : "+str(image_path)+"\nFailed to identify face !!")
f = open("./files/confusion_matrix.pickle", "wb")
f.write(pickle.dumps([actual,predicted]))
f.close()