import cv2, os, glob, sys, imutils, dlib
from pathlib import Path
from mtcnn.mtcnn import MTCNN
from imutils.face_utils import FaceAligner
from files.utilities import calcBoxArea
from files.utilities import printProgressBar

detector = MTCNN()
fa = FaceAligner(dlib.shape_predictor("./files/shape.dat"), desiredFaceWidth=100)

if os.path.isdir("./dataset/1_extracted"):
  rmtree("./dataset/1_extracted")
os.mkdir("./dataset/1_extracted")  
os.makedirs("./dataset/1_extracted/sober") 
os.makedirs("./dataset/1_extracted/drunk") 

frameskip = 100
facecount = 1
states = ["drunk","sober"]
for state in states:
	files = sorted(glob.glob("./dataset/0_videos/"+state+"/"))
	print("Reading "+state+" Videos")
	printProgressBar(0, len(files), prefix = "Processing "+state+" Images", suffix = "Complete", length = 50)	
	for i, file in enumerate(files):
		cam = cv2.VideoCapture(file)
		currentframe = 1
		while(True): 
			ret,image = cam.read() 
			if ret: 
				if currentframe==frameskip:
					currentframe = 1
					result = detector.detect_faces(image)
					if result:
						result.sort(key = calcBoxArea, reverse = True)
						for face in result:
							(x, y, w, h) = (face['box'][0],face['box'][1],face['box'][2],face['box'][3])
							faceAligned = fa.align(image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dlib.rectangle(x,y,w+x,h+y))
							cv2.imwrite("./dataset/1_extracted/"+str(state)+"/"+str(state) +"_"+ str(facecount) + ".png", faceAligned)
							print("State : "+str(state)+" FaceNo : "+str(facecount)+" Video : "+str(vid))
							facecount = facecount +1
				else:
					currentframe = currentframe +1
			else: 
				break
		cam.release() 
		cv2.destroyAllWindows()
		printProgressBar(i, len(files), prefix = "Processing "+state+" Images", suffix = "Complete", length = 50)	
