from imutils import paths
import face_recognition, pickle, cv2, os
from files.utilities import printProgressBar

states = ["drunk", "sober"]
for state in states:
	print("Encoding "+state+" Images")
	imagePaths = list(paths.list_images("./dataset/1_extracted/"+state))
	data = []
	printProgressBar(0, len(imagePaths), prefix = "Encoding "+state+" Images", suffix = "Complete", length = 50)	
	for (i, imagePath) in enumerate(imagePaths):
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb,model="cnn")
		encodings = face_recognition.face_encodings(rgb, boxes)
		d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
		data.extend(d)
		printProgressBar(i, len(imagePaths), prefix = "Encoding "+state+" Images", suffix = "Complete", length = 50)	
	print("Serializing "+state+" Encodings...")
	f = open("./files/"+state+".pickle", "wb")
	f.write(pickle.dumps(data))
	f.close()
	print("Pickle File Generated at ./files/"+state+".pickle")