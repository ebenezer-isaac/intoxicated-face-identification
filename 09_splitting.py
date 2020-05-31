 
import glob, os
from shutil import copyfile, rmtree
from random import randint
from files.utilities import printProgressBar

train_percentage = 70

files = sorted(glob.glob('./dataset/6_combined/*'))
max_subject = 0
no_of_files = len(files)
for file in files:
	file = file.split("/")
	file = file[len(file)-1]
	if max_subject< int(file[:3]):
		max_subject = int(file[:3])
test_count = round(((no_of_files/100)*(100-train_percentage)))
test_list = []
test_filelist = []
while len(test_filelist)<test_count:
	rand = randint(1,max_subject+1)
	if rand in test_list:
		pass
	else:
		rand = "{:03d}".format(rand)
		test_list.append(rand)
		for file in files:
			if file.find(rand)>=0:
				test_filelist.append(file)

print(
	"Random Test Subjects :",len(test_list),"\n"
	"Test Percentage      :",100-train_percentage,"\n"
	"Train Percentage     :",train_percentage,"\n"
	"Test Images          :",len(test_filelist),"\n"
	"Train Images         :",no_of_files-len(test_filelist),"\n"
	"Total Files          :",no_of_files)

if os.path.isdir("./dataset/7_main/"):
	rmtree("./dataset/7_main/")
os.mkdir("./dataset/7_main")   
os.mkdir("./dataset/7_main/test")   
os.mkdir("./dataset/7_main/test/sober") 
os.mkdir("./dataset/7_main/test/drunk") 
os.mkdir("./dataset/7_main/train")   
os.mkdir("./dataset/7_main/train/sober") 
os.mkdir("./dataset/7_main/train/drunk") 

printProgressBar(0, no_of_files, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, file in enumerate(files):
	if file in test_filelist:
		if (file.find('sober') != -1):
			copyfile(file, "./dataset/7_main/test/sober/"+file.replace("./dataset/6_combined/",""))
		else:
			copyfile(file, "./dataset/7_main/test/drunk/"+file.replace("./dataset/6_combined/",""))
	else:
		if (file.find('sober') != -1):
			copyfile(file, "./dataset/7_main/train/sober/"+file.replace("./dataset/6_combined/",""))
		else:
			copyfile(file, "./dataset/7_main/train/drunk/"+file.replace("./dataset/6_combined/",""))
	printProgressBar(i + 1, no_of_files, prefix = 'Progress:', suffix = 'Complete', length = 50)