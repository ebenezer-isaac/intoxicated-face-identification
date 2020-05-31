import os
from shutil import rmtree

dir = "./dataset/6_combined/"
if os.path.isdir(dir):
	rmtree(dir)
os.mkdir(dir)