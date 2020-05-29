 
import glob, os, hashlib
from shutil import copyfile, rmtree
from random import randint
from files.utilities import printProgressBar

directories = ["./dataset/7_main/train/sober/*","./dataset/7_main/train/drunk/*","./dataset/7_main/test/sober/* ","./dataset/7_main/test/drunk/* "]
for directory in directories:
	files = sorted(glob.glob(directory.strip()))
	printProgressBar(0, len(files), prefix = directory+" : ", suffix = 'Complete', length = 50)
	for i, file in enumerate(files):
		filename = file.split("/")
		filename = filename[len(filename)-1]
		new_name = hashlib.sha256((hashlib.sha256(filename.encode()).hexdigest()).encode()).hexdigest()
		os.rename(file, file.replace(filename,"{}.png".format(new_name)))
		printProgressBar(i, len(files), prefix = directory+" : ", suffix = 'Complete', length = 50)
	printProgressBar(len(files), len(files), prefix = directory+" : ", suffix = 'Complete', length = 50)
