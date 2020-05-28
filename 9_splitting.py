 
import glob, os
from shutil import copyfile, rmtree
from random import randint
from utilities import printProgressBar

train_percentage = 85


test_percentage = 100-train_percentage
test_count = round(1.27*test_percentage)
test_list = []
while len(test_list)<test_count:
	rand = randint(1,128)
	if rand in test_list:
		pass
	else:
		test_list.append(rand)
print("Random Test Subjects :",test_list)
os.chdir("./data/dataset")
if os.path.isdir("../split"):
	rmtree("../split/")
os.mkdir("../split")   
os.mkdir("../split/test")   
os.mkdir("../split/test/sober") 
os.mkdir("../split/test/drunk") 
os.mkdir("../split/train")   
os.mkdir("../split/train/sober") 
os.mkdir("../split/train/drunk") 
files = sorted(glob.glob('*'))
no_of_files = len(files)
printProgressBar(0, no_of_files, prefix = 'Progress:', suffix = 'Complete', length = 50)
for i, file in enumerate(files):
	subject = file.split("_")
	subject = int(subject[0])
	if subject in test_list:
		if (file.find('sober') != -1):
			copyfile(file, "../split/test/sober/"+file)
		else:
			copyfile(file, "../split/test/drunk/"+file)
	else:
		if (file.find('sober') != -1):
			copyfile(file, "../split/train/sober/"+file)
		else:
			copyfile(file, "../split/train/drunk/"+file)
	printProgressBar(i + 1, no_of_files, prefix = 'Progress:', suffix = 'Complete', length = 50)
