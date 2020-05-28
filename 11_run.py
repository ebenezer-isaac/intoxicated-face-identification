from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import optimizers
from skimage import io
from skimage.transform import resize
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import random, glob, coremltools, os, cv2, imutils, dlib, requests
from imutils.face_utils import FaceAligner
from mtcnn.mtcnn import MTCNN
from utilities import calcBoxArea

nImageRows = 100
nImageCols = 100
nChannels = 3


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
model.load_weights('model3.h5')

binary_dict = {0: "Sober", 1: "Drunk"}
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape.dat")
fa = FaceAligner(predictor, desiredFaceWidth=100)
mtcnn = MTCNN()
# start the webcam feed
while True:
    url = "http://172.21.0.3:8080/shot.jpg"
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)
    frame =  cv2.resize(img,(240,427),interpolation = cv2.INTER_CUBIC)
    result = mtcnn.detect_faces(frame)
    if result:
        result.sort(key = calcBoxArea, reverse = True)
        (x, y, w, h) = (result[0]['box'][0],result[0]['box'][1],result[0]['box'][2],result[0]['box'][3])
        faceAligned = fa.align(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dlib.rectangle(x,y,w+x,h+y))
        image = np.expand_dims(faceAligned, axis=0)
        prediction = model.predict(image)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, binary_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(frame,(240,427),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()