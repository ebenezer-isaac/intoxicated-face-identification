from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob, cv2, dlib, pickle, os
from imutils.face_utils import FaceAligner
from mtcnn.mtcnn import MTCNN
from files.utilities import calcBoxArea
from files.utilities import printProgressBar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

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

error_count = 0
for state in states:
	images = glob.glob("./dataset/7_main/test/"+str(state)+"/*")
	printProgressBar(0, len(images), prefix = "Predicting "+state+" Images", suffix = "Complete", length = 50)
	for index, image_path in enumerate(images):
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
		else:
			error_count = error_count +1
		printProgressBar(index, len(images), prefix = "Predicting "+state+" Images", suffix = "Complete", length = 50)
	print("")
print("Errors :",error_count)
f = open("./files/tensorflow_prediction.pickle", "wb")
f.write(pickle.dumps([actual,predicted]))
f.close()
print("Predictions saved to ./files/tensorflow_prediction.pickle")