 
import glob, os
from shutil import copyfile, rmtree
from random import randint
dir = "./data/dataset/"
if os.path.isdir(dir):
	rmtree(dir)
os.mkdir(dir)
states = ["sober_00","sober_01","drunk_02","drunk_03"]
for subject in range(1,54):
	for state in states:
		file_list = []
		files = sorted(glob.glob("../v2/data/3_cleaned/{:02d}_{}*".format(subject,state)))
		if files:
			print("../v2/data/3_cleaned/{:02d}_{}*".format(subject,state)+" : ",len(files))
			limit = 9
			if len(files)<limit:
				limit = len(files)
			while len(file_list)<limit:
				rand = randint(0,len(files)-1)
				if files[rand] in file_list:
					pass
				else:
					file_list.append(files[rand])
			dir_state = "sober"
			if state.find("sober")==-1:
				dir_state = "drunk"
			dir = "./data/dataset/"
			#if os.path.isdir(dir+dir_state+"_"+str(subject)):
			#	rmtree(dir+dir_state+"_"+str(subject))
			#os.mkdir(dir+dir_state+"_"+str(subject))
			for file in file_list:
				filename = file.split("/")
				filename = filename[len(filename)-1]
				print(dir+filename)
				copyfile(file, dir+filename)
		