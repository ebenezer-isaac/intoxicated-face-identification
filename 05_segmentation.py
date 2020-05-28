import cv2, os, glob, sys, random, itertools, dlib
from utilities import printProgressBar
from shutil import rmtree
import zipfile

if os.path.isdir('./dataset/3_wine'):
	rmtree('./dataset/3_wine')
os.mkdir('./dataset/3_wine')   

with zipfile.ZipFile("./files/wine_project.zip" 'r') as zip_ref:
    zip_ref.extractall('./dataset/3_wine')

count = 0
files = glob.glob('./dataset/3_wine/*')
printProgressBar(0, len(files)*4, prefix = 'Segmenting Images:', suffix = 'Complete', length = 50)
image_count = 0
for file in files:
	count = count+1
	img = cv2.imread(file, cv2.IMREAD_COLOR)
	h, w = img.shape[:2]
	cropped=[]
	cropped.append(img[:h//2, :w//2])
	cropped.append(img[:h//2, w//2:])
	cropped.append(img[h//2:, :w//2])
	cropped.append(img[h//2:, w//2:])
	for index, image in enumerate(cropped):
		state=""
		if index<2:
			state = "sober"
		else:
			state = "drunk"
		image_count = image_count +1
		cv2.imwrite('../4_segmented/{:03d}_{}_{:02d}.png'.format(count, state, index), image)
		printProgressBar(image_count, len(files)*4, prefix = 'Segmenting Images:', suffix = 'Complete', length = 50)
