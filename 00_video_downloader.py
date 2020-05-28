from __future__ import unicode_literals
import os, youtube_dl
try: 
	if not os.path.exists("dataset"): 
		os.makedirs("dataset") 
	if not os.path.exists("dataset/0_videos"): 
		os.makedirs("dataset/0_videos") 
	if not os.path.exists("dataset/0_videos/sober"): 
		os.makedirs("dataset/0_videos/sober") 
	if not os.path.exists("dataset/0_videos/drunk"): 
		os.makedirs("dataset/0_videos/drunk") 
except OSError: 
	print ("Error: Creating directory of data")
os.chdir("./dataset/0_videos/")
states = ["drunk", "sober"]
for state in states:
	os.chdir("./"+str(state)+"/")
	links=open("../../../files/"+str(state)+".txt","r") 
	for i in links: 
		try:
			ydl_opts = {}
			with youtube_dl.YoutubeDL(ydl_opts) as ydl:
				ydl.download([i])
				print(i)
		except Exception as e:
			pass
	os.chdir("../")