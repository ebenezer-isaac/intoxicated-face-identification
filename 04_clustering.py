from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import pickle, cv2, os, glob
from shutil import copyfile, rmtree

if os.path.isdir("./dataset/2_cluster"):
	rmtree("./dataset/2_cluster")
os.mkdir("./dataset/2_cluster") 

files = glob.glob("./files/*.pickle")
for file in files:
	print("Reading "+file)
	iteration = file.split("/")
	iteration = iteration[len(iteration)-1].replace(".pickle","")
	data = pickle.loads(open(file, "rb").read())
	data = np.array(data)
	encodings = [d["encoding"] for d in data]
	print("\nClustering Images in "+iteration)
	clt = DBSCAN(metric="euclidean", n_jobs=int("-1"))
	clt.fit(encodings)
	labelIDs = np.unique(clt.labels_)
	numUniqueFaces = len(np.where(labelIDs > -1)[0])
	print("Unique Classes Found: {}".format(numUniqueFaces))
	for labelID in labelIDs:
		if labelID >= 0:
			print("Copying Face Class ID: {}".format(labelID),"./dataset/2_cluster/"+iteration+"_"+str(labelID))
			if os.path.isdir("./dataset/2_cluster/"+iteration+"_"+str(labelID)):
				rmtree("./dataset/2_cluster/"+iteration+"_"+str(labelID))
			os.mkdir("./dataset/2_cluster/"+iteration+"_"+str(labelID)) 
			idxs = np.where(clt.labels_ == labelID)[0]
			faces = []
			for i in idxs:
				filename = data[i]["imagePath"].split("/")
				filename = filename[len(filename)-1]
				copyfile(data[i]["imagePath"], "./dataset/2_cluster/"+iteration+"_"+str(labelID)+"/"+str(filename))