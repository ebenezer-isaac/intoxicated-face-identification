from sklearn.linear_model import LogisticRegression
import numpy as np
import cv2, dlib, requests, pickle
from imutils.face_utils import FaceAligner
from mtcnn.mtcnn import MTCNN
from files.utilities import calcBoxArea

logisticRegr = LogisticRegression(max_iter=1000,verbose=1)
print("Reading Files")
logisticRegr = pickle.load(open("./files/loginstic_regression.pickle", 'rb'))
print("Files Loaded")

states=["sober","drunk"]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./files/shape.dat")
fa = FaceAligner(predictor, desiredFaceWidth=100)
mtcnn = MTCNN()
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
        prediction = states[int(logisticRegr.predict(image.reshape(1,-1)))]
        cv2.putText(frame, prediction.title(), (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(frame,(240,427),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()