import pandas as pd
import librosa.display
import numpy as np
from numpy.lib import stride_tricks
import os
from numpy import save
from numpy import savetxt
from numpy import load
from pandas import DataFrame
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import load_model


def gru_model(filename,audio_data,face_data,video_data):

	x_val_audio=np.array(audio_data)
	x_val_face=np.array(face_data)
	x_val_video=np.array(video_data)
	print("shapes")
	print(x_val_audio.shape,x_val_face.shape,x_val_video.shape)
	model=load_model("/home/ubuntu/Desktop/Sowmya/AffWild2/feat_new/affew_model.h5")
	predicted_output = model.predict([x_val_video,x_val_face,x_val_audio])
	#print(predicted_output.shape)
	
	text_file_gen(predicted_output,filename)

def text_file_gen(predicted_output,filename):

	data = predicted_output

	print(filename)
	filename="./gru_text_files/"+str(filename)+".txt"
	print(filename)

	# Write the array to disk
	with open(filename, 'w') as outfile:

		for i in range(len(data)):
			
		
			if i==0:
				pre_data=data[i]
				pre_data=pre_data[:-5,:]
				d_slice=desplitting(data[i],data[i+1])
				data_slice=np.vstack([pre_data,d_slice])
			elif i==len(data)-1:
				pre_data=data[i]
				pre_data=pre_data[5:,:]
				print(pre_data.shape)
				data_slice=pre_data
			else:
				pre_data=data[i]
				pre_data=pre_data[:-5,:]
				pre_data=pre_data[5:,:]
				d_slice=desplitting(data[i],data[i+1])
				data_slice=np.vstack([pre_data,d_slice])
			np.savetxt(outfile, data_slice, fmt='%-7.20f')
	

def desplitting(data1,data2):
	append1=data1[10:, :]
	append2=data2[:-10, :]
	avg_res=np.mean( np.array([ append1, append2 ]), axis=0 )
	return(avg_res)

x_val_audio =x_val_audio = np.load('/new_test_data.npy', allow_pickle=True)
x_val_video = np.load('/test_data_video.npy', allow_pickle=True)
x_val_face=np.load('/test_feat_face.npy', allow_pickle=True)


if __name__ == '__main__':
	for i in range(len(x_val_audio)):
		for j in range(len(x_val_face)):
			for k in range(len(x_val_video)):
				if(x_val_audio[i][0]==x_val_face[j][0][:-4]==x_val_video[k][0]):
					filename=np.array(x_val_audio[i][0])
					#print("filename",filename)
					audio_data=np.array(x_val_audio[i][1])
				face_data=np.array(x_val_face[j][1])	
				video_data=np.array(x_val_video[k][1])
				gru_model(filename,audio_data,face_data,video_data)

			

		
	
		
















