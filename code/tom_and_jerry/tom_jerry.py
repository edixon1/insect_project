import numpy as np
import cv2 #video capturing
import math
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
from keras.utils import np_utils
from skimage.transform import resize
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

count = 0
videoFile = "Tom and jerry.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5)

#x=1
def get_frames():
	count = 0
	videoFile = "Tom and jerry.mp4"
	cap = cv2.VideoCapture(videoFile)
	frameRate = cap.get(5)
	while(cap.isOpened()):
		frameId = cap.get(1) #current frame number
		ret, frame = cap.read()
		if (ret != True):
			break
		if (frameId % math.floor(frameRate) == 0):
			filename = "frame%d.jpg" % count;count+=1
			cv2.imwrite(filename,frame)
	cap.release()
	print("Done!")


data = pd.read_csv('mapping.csv')
X = []
for img_name in data.Image_ID:
	img = plt.imread(''+img_name)
	X.append(img)
X = np.array(X)

y = data.Class
dummy_y = np_utils.to_categorical(y)

image = []
for i in range(0,X.shape[0]):
	a = resize(X[i], preserve_range=True,output_shape=(224,224)).astype(int)
	image.append(a)
X = np.array(image)


X = preprocess_input(X,mode='tf')

# img = plt.imread('frame0.jpg')
# plt.imshow(img)
# plt.show()