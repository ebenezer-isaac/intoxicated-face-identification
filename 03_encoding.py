from imutils import paths
import face_recognition, pickle, cv2, os, glob
from files.utilities import printProgressBar

iteration_length = 500

states = ["drunk", "sober"]
for state in states:
	files = glob.glob("./files/"+state+"*.pickle")
	for file in files:
		os.remove(file)
	print("Encoding "+state+" Images")
	imagePaths = list(paths.list_images("./dataset/1_extracted/"+state))
	iterations = round(len(imagePaths)/iteration_length)
	while iterations*iteration_length < len(imagePaths):
		iterations = iterations+1
	counter = 0
	print("Iterations :",iterations, "Iterations Length :", iteration_length,"Total Images :",len(imagePaths))
	for iteration in range(1,iterations+1):
		data = []
		print("Iteration :",iteration, "[{}:{}]".format(counter,iteration*iteration_length+1))
		subset = imagePaths[counter:iteration*iteration_length+1]
		counter = iteration*iteration_length+1
		printProgressBar(0, len(subset), prefix = "Encoding "+state+" Images", suffix = "Complete", length = 50)	
		for (i, imagePath) in enumerate(subset):
			image = cv2.imread(imagePath)
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			boxes = face_recognition.face_locations(rgb,model="cnn")
			encodings = face_recognition.face_encodings(rgb, boxes)
			d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
			data.extend(d)
			printProgressBar(i, len(subset), prefix = "Encoding "+state+" Images", suffix = "Complete", length = 50)
		printProgressBar(len(subset), len(subset), prefix = "Encoding "+state+" Images", suffix = "Complete", length = 50)
		print("Serializing "+state+" Encodings...")
		f = open("./files/"+state+"_"+str(iteration)+".pickle", "wb")
		f.write(pickle.dumps(data))
		f.close()
		print("Pickle File Generated at ./files/"+state+"_"+str(iteration)+".pickle")
	print("")
	print("")