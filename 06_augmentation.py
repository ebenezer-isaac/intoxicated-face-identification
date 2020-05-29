import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2, itertools, os, glob, random, dlib
from functools import reduce
from files.utilities import printProgressBar, calcBoxArea
from shutil import rmtree
from mtcnn.mtcnn import MTCNN
from imutils.face_utils import FaceAligner

mtcnn = MTCNN()
fa = FaceAligner(dlib.shape_predictor("./files/shape.dat"), desiredFaceWidth=100)

limit = 20000

meta = {'noop': iaa.Noop(), 'shuffle': iaa.ChannelShuffle(p=1.0)}
arithmetic = {
	'add-45': iaa.Add(value=-45),
	'addp-': iaa.Add(value=(-35, -15), per_channel=True), 'addp+': iaa.Add(value=(15, 35), per_channel=True),
	'addGN': iaa.AdditiveGaussianNoise(scale=0.10 * 255),
	'addGNp': iaa.AdditiveGaussianNoise(scale=0.10 * 255, per_channel=True),
	'addLN': iaa.AdditiveLaplaceNoise(scale=0.10 * 255),
	'addPN': iaa.AdditivePoissonNoise(lam=16.00),
	'mul-': iaa.Multiply(mul=0.50), 'mul+': iaa.Multiply(mul=1.50),
	'mulp-': iaa.Multiply(mul=0.50, per_channel=True), 'mulp+': iaa.Multiply(mul=1.50, per_channel=True),
	'jpeg': iaa.JpegCompression(compression=62), 'jpeg+': iaa.JpegCompression(compression=75),
	'jpeg++': iaa.JpegCompression(compression=87)
}
blur = {
	'GBlur': iaa.GaussianBlur(sigma=1.00),
	'ABlur': iaa.AverageBlur(k=3), 'MBlur': iaa.MedianBlur(k=3),
	'BBlur': iaa.BilateralBlur(sigma_color=250, sigma_space=250, d=5),
	'MoBlur': iaa.MotionBlur(angle=0, k=7), 'MoBlurAng': iaa.MotionBlur(angle=144, k=5)
}
color = {
	'ATHAS-': iaa.AddToHueAndSaturation(value=-45), 'ATHAS+': iaa.AddToHueAndSaturation(value=45),
	'Gray': iaa.Grayscale(alpha=0.2)
}
contrast = {
	'GContrast-': iaa.GammaContrast(gamma=0.81),
	'GContrast+': iaa.GammaContrast(gamma=1.44), 'SContrast': iaa.SigmoidContrast(cutoff=0.5, gain=10),
	'LContrast': iaa.LogContrast(gain=0.88), 'LiContrast': iaa.LinearContrast(alpha=1.38)
}
weather = {'Clouds': iaa.Clouds()}
effects = {**meta, **arithmetic, **blur, **color, **contrast, **weather}

if os.path.isdir("./dataset/5_augmented"):
	rmtree("./dataset/5_augmented")
os.mkdir("./dataset/5_augmented")

count = 0
printProgressBar(0, limit + 1000, prefix = 'Segmenting Images:', suffix = 'Complete', length = 50)
for i in range(2):
	combs = list(itertools.combinations(effects.keys(), i + 1))
	combs = random.sample(combs, len(combs))
	for aug in combs:
		if count > limit:
			printProgressBar(limit + 1000, limit + 1000, prefix = 'Segmenting Images:', suffix = 'Complete', length = 50)
			break
		aug_effects = list(map(lambda x: effects[x], aug))
		if not len(aug_effects) - 1:
			seq = aug_effects[0].to_deterministic()
		else:
			seq = iaa.Sequential(aug_effects).to_deterministic()
		for file in glob.glob('./dataset/4_segmented/*'):
			img = cv2.imread(file, cv2.IMREAD_COLOR)
			aug_img = seq.augment_image(img)
			ext = reduce((lambda x, y: str(x) + '_' + str(y)), aug)
			result = mtcnn.detect_faces(aug_img)
			if result:
				result.sort(key = calcBoxArea, reverse = True)
				(x, y, w, h) = (result[0]['box'][0],result[0]['box'][1],result[0]['box'][2],result[0]['box'][3])
				faceAligned = fa.align(aug_img, cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY), dlib.rectangle(x,y,w+x,h+y))
				cv2.imwrite('./dataset/5_augmented/{}_{}.png'.format(file.replace("./dataset/4_segmented/",""),ext), faceAligned)
				count = count+1
			printProgressBar(count, limit + 1000, prefix = 'Segmenting Images:', suffix = 'Complete', length = 50)
			count += 1