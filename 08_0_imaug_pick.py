 
import glob
from shutil import copyfile
from random import randint
from files.utilities import printProgressBar

main_limit = 50

dir = "./dataset/6_combined/"
states = ["sober_00","sober_01","drunk_02","drunk_03"]
counter = 0
printProgressBar(counter, 53*4*main_limit, prefix = 'Progress:', suffix = 'Complete', length = 50)
for subject in range(1,54):
	for state in states:
		file_list = []
		files = sorted(glob.glob("./dataset/5_augmented/{:03d}_{}*".format(subject,state)))
		if files:
			limit = main_limit
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
			for file in file_list:
				filename = file.split("/")
				filename = filename[len(filename)-1]
				copyfile(file, dir+filename)
				counter = counter +1
				printProgressBar(counter, 53*4*main_limit, prefix = 'Progress:', suffix = 'Complete', length = 50)
	counter = subject*4*main_limit
	printProgressBar(counter, 53*4*main_limit, prefix = 'Progress:', suffix = 'Complete', length = 50)
		