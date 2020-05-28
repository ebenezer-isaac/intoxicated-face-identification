import glob, os
from shutil import copyfile, rmtree
from random import randint
folders = sorted(glob.glob("./dataset/2_cluster/*"))
count = 1
for folder in folders:
	files = sorted(glob.glob(folder+"/*"))
	if files:
		print(folder+"/ : ",len(files))
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
		dir = "./dataset/3_combined/"
		for file in file_list:
			subject = folder.split("_")
			subject = int(subject[len(subject)-1])
			filename = dir+str(subject)+"_"+dir_state+"_"+str(count)
			print(filename)
			copyfile(file, filename+".png")
			count = count+1