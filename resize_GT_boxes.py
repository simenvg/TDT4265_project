import pickle
import cv2
import os


folder = 'resized_images'

folder_path = '/home/simenvg/environments/my_env/datasyn/project/' + folder


images_GT = pickle.load(open(os.path.join(folder_path,'ground_truth.txt'), "rb"))

images_GT_resized = {}

i = 0

for keys, values in images_GT.items():
	resized_boxes = []
	i += 1
	for box in values:
		resized_point_1 = (int(box[0][0] * 0.5), int(box[0][1] * 0.5)) 
		resized_point_2 = (int(box[1][0] * 0.5), int(box[1][1] * 0.5))
		resized_box = [resized_point_1, resized_point_2]
		resized_boxes.append(resized_box)
		print('BOX: ', box)
		print('Resized: ', resized_box)
	image_name = 'resized_' + keys
	images_GT_resized[image_name] = resized_boxes
print('i = ', i)
pickle.dump(images_GT_resized, open(os.path.join(folder_path,'ground_truth_resized.txt'), "wb"))





