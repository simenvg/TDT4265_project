import cv2
import os




folder_path = '/home/simenvg/environments/my_env/datasyn/project/JPEGImages'


for filename in os.listdir(folder_path):
	if filename == 'ground_truth.txt':
		continue
	img = cv2.imread(os.path.join(folder_path,filename))
	img_resized = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	cv2.imwrite("resized_" + filename, img_resized)