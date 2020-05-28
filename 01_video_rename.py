import os, glob 
os.chdir("./dataset/0_videos/")
states = ["drunk","sober"]
for state in states:
	files = sorted(glob.glob("./"+state+"/*"))
	counter = 1
	for file in files:
		path = file.rsplit("/", 1) 
		filename = path[0]+"/"+str(state)+"_"+"{:02d}".format(counter)+".mp4"
		print(file, "   ", filename)
		os.rename(file,filename)
		counter = counter+1