import glob, os
from shutil import copyfile, rmtree
from random import randint
from files.utilities import printProgressBar

folders = sorted(glob.glob("./dataset/2_cluster/*"))
count = 1
printProgressBar(count,len(folders)*8, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, folder in enumerate(folders):
	files = sorted(glob.glob(folder+"/*"))
	if files:
		limit = 8
		if len(files)<limit:
			limit = len(files)
		file_list = []
		while len(file_list)<limit:
			rand = randint(0,len(files)-1)
			if files[rand] in file_list:
				pass
			else:
				file_list.append(files[rand])
		dir_state = "sober_00"
		if folder.find("sober")==-1:
			dir_state = "drunk_03"
		dir = "./dataset/6_combined/"
		for file in file_list:
			subject = folder.split("_")
			subject = int(subject[len(subject)-1])+53
			filename = dir+"{:03d}".format(int(subject))+"_"+dir_state+"_"+str(count)
			copyfile(file, filename+".png")
			count = count+1
			printProgressBar(count,len(folders)*8, prefix = 'Progress:', suffix = 'Complete', length = 50)
	count = i*8
	printProgressBar(count,len(folders)*8, prefix = 'Progress:', suffix = 'Complete', length = 50)
printProgressBar(len(folders)*8,len(folders)*8, prefix = 'Progress:', suffix = 'Complete', length = 50)