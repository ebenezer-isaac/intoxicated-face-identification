from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle, cv2, os
from shutil import copyfile, rmtree

if os.path.isdir("./dataset/2_cluster"):
	rmtree("./dataset/2_cluster")
os.mkdir("./dataset/2_cluster") 

states = ["drunk", "sober"]
for state in states:
	print("Reading "+state+" Encoding")
	data = pickle.loads(open("./files/"+state+".pickle", "rb").read())
	data = np.array(data)
	encodings = [d["encoding"] for d in data]
	print("Clustering Images\n This may take some time")
	clt = DBSCAN(metric="euclidean", n_jobs="-1")
	clt.fit(encodings)
	labelIDs = np.unique(clt.labels_)
	numUniqueFaces = len(np.where(labelIDs > -1)[0])
	print("\nUnique Classes Found: {}".format(numUniqueFaces))
	for labelID in labelIDs:
		if labelID >= 0:
			print("Copying Face Class ID: {}".format(labelID))
			if os.path.isdir("./dataset/2_cluster/"state+"_"+str(labelID)):
				rmtree("./dataset/2_cluster/"state+"_"+str(labelID))
			os.mkdir("./dataset/2_cluster/"state+"_"+str(labelID)) 
			idxs = np.where(clt.labels_ == labelID)[0]
			faces = []
			for i in idxs:
				filename = data[i]["imagePath"].split("/")
				filename = filename[len(filename)-1]
				copyfile(data[i]["imagePath"], "./dataset/2_cluster/"state+"_"+str(labelID)+"/"+str(filename))
