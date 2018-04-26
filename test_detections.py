import pickle
import cv2
import os
from scipy.spatial import distance
import copy
from matplotlib import pyplot as plt





folder_path = '/home/simenvg/environments/my_env/datasyn/project/JPEGImages'
folder_path_resized = '/home/simenvg/environments/my_env/datasyn/project/resized_images'


BLUE = (255,0,0)
RED = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,255,0)


images_YOLO = pickle.load(open(os.path.join(folder_path,'YOLO.txt'), "rb"))
images_GT = pickle.load(open(os.path.join(folder_path,'ground_truth.txt'), "rb"))

images_YOLO_resized = pickle.load(open(os.path.join(folder_path_resized,'YOLO.txt'), "rb"))
images_GT_resized = pickle.load(open(os.path.join(folder_path_resized,'ground_truth_resized.txt'), "rb"))

filenames = []
filenames_resized = []

for keys, values in images_GT.items():
	filenames.append(keys)

for keys, values in images_GT_resized.items():
	filenames_resized.append(keys)


images_YOLO_conf_over_25 = {}

for key, value in images_YOLO.items():
	boxes = []
	scores = []
	for i in range(len(value[0])):
		if value[1][i] >= 0.25:
			boxes.append(value[0][i])
			scores.append(value[1][i])
	images_YOLO_conf_over_25[key] = (boxes, scores)




def intersect_area(box_1, box_2):  # returns None if rectangles don't intersect
	dx = min(max(box_1[1][0], box_1[0][0]), box_2[1][0]) - max(min(box_1[1][0], box_1[0][0]), box_2[0][0])
	dy = min(max(box_1[1][1], box_1[0][1]), box_2[1][1]) - max(min(box_1[1][1], box_1[0][1]), box_2[0][1])
	#print('dx: ', dx, '    dy: ', dy)
	if (dx>=0) and (dy>=0):
		return dx*dy
	else:
		return -1


def intersect_over_union(box_1, box_2):
	#print('box_1: ', box_1, '   box_2: ', box_2)
	area_box_1 = abs((box_1[1][0] - box_1[0][0]) * (box_1[1][1] - box_1[0][1]))
	area_box_2 = abs((box_2[1][0] - box_2[0][0]) * (box_2[1][1] - box_2[0][1]))
	intersection = intersect_area(box_1, box_2)
	#print('area_box_1: ', area_box_1, '   area_box_2: ', area_box_2, '  intersection: ', intersection)
	if intersection == -1:
		return -1
	else:
		return intersection / (area_box_1 + area_box_2 - intersection)


def validated_detected_objects(detected_boxes, GT_boxes):
	approved_boxes = []
	temp_detected_boxes = copy.copy(detected_boxes)
	for GT_box in GT_boxes:
		if len(temp_detected_boxes) > 0:
			found_box = False
			best_iou = intersect_over_union(GT_box, temp_detected_boxes[0])
			if best_iou >= 0.5:
				found_box = True
			best_box = temp_detected_boxes[0]
			for i in range(1, len(temp_detected_boxes)):
				iou = intersect_over_union(GT_box, temp_detected_boxes[i])
				if iou >= 0.5 and iou > best_iou:
					found_box = True
					best_box = temp_detected_boxes[i]
					best_iou = iou
			if found_box:
				approved_boxes.append(best_box)
				temp_detected_boxes.remove(best_box)
	return approved_boxes

def get_box_center(box):
	x = abs((box[0][0] + box[1][0])/2)
	y = abs((box[0][1] + box[1][1])/2)
	return (x,y)


def euc_dist(point_1, point_2):
	return distance.euclidean(point_1, point_2)


def get_precision(filenames, GT_boxes, detected_boxes):
	sum_detected_objects = 0
	sum_true_positives = 0
	for image in filenames:
		sum_detected_objects += len(detected_boxes[image])
		sum_true_positives += len(validated_detected_objects(detected_boxes[image], GT_boxes[image]))
	if sum_detected_objects == 0:
		return -1
	return sum_true_positives / sum_detected_objects

def get_recall(filenames, GT_boxes, detected_boxes):
	sum_GT_boxes = 0
	sum_true_positives = 0
	for image in filenames:
		sum_GT_boxes += len(GT_boxes[image])
		sum_true_positives += len(validated_detected_objects(detected_boxes[image], GT_boxes[image]))
	return sum_true_positives / sum_GT_boxes


def boxes_based_on_score(filenames, detected_boxes, conf_level):
	images = {}
	for image in filenames:
		boxes = []
		for i in range(len(detected_boxes[image][0])):
			if float(detected_boxes[image][1][i]) > conf_level:
				boxes.append(detected_boxes[image][0][i])
		images[image] = boxes
	return images


def plot_precision_recall(yolo_prec_recall, yolo_prec_recall_resized):
	fig_1, ax_1 = plt.subplots()
	for i in range(len(yolo_prec_recall[0])):
		if yolo_prec_recall[0][i] != -1:
			ax_1.plot(yolo_prec_recall[0][i], yolo_prec_recall[1][i], 'o', color='red', label='YOLO')#, AP: ' + str(round(get_average_precision(yolo_prec_recall),2)))
	for i in range(len(yolo_prec_recall[0])):
		if yolo_prec_recall_resized[0][i] != -1:
			ax_1.plot(yolo_prec_recall_resized[0][i], yolo_prec_recall[1][i], 'o', color='blue', label='YOLO resized images')
	ax_1.set_title('Precision/Recall')
	plt.grid()
	plt.legend()
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.show()



thresholds = [0.999, 0.99]

a = 0.95
while a >= 0.01:
	thresholds.append(a)
	a -= 0.01

thresholds.extend([0.001, 0.000001, 0.0000000001])


def generate_recall_and_precisions(filenames, GT_boxes, detected_boxes, thresholds):
	recalls = []
	precisions = []
	for elem in thresholds:
		recalls.append(get_recall(filenames, GT_boxes, boxes_based_on_score(filenames, detected_boxes, elem)))
		precisions.append(get_precision(filenames, GT_boxes, boxes_based_on_score(filenames, detected_boxes, elem)))

	return (precisions, recalls)


def get_average_precision(prec_recall):
	precisions = prec_recall[0]
	sum_precisions = 0
	for elem in precisions:
		sum_precisions += elem
	return sum_precisions/len(precisions)



yolo_prec_recall = generate_recall_and_precisions(filenames, images_GT, images_YOLO, thresholds)
yolo_prec_recall_resized = generate_recall_and_precisions(filenames_resized, images_GT_resized, images_YOLO_resized, thresholds)


print('YOLO AP: ', get_average_precision(yolo_prec_recall))

plot_precision_recall(yolo_prec_recall, yolo_prec_recall_resized)

sum_GT_boxes = 0
sum_validated_boxes = 0
for image in filenames:
	validated_boxes = validated_detected_objects(images_YOLO_conf_over_25[image][0], images_GT[image])
	sum_validated_boxes += len(validated_boxes)
	sum_GT_boxes += len(images_GT[image])


print('Detected ', sum_validated_boxes, ' out of ', sum_GT_boxes, ' boats,   ', sum_validated_boxes * 100/ sum_GT_boxes, '  %')





def draw_boxes(image_name, GT_boxes, detected_boxes):
	validated_boxes = validated_detected_objects(detected_boxes, GT_boxes)
	img = cv2.imread(os.path.join(folder_path,image_name))
	for box in detected_boxes:
		cv2.rectangle(img, box[0], box[1], RED, 1)
	for box in validated_boxes:
		cv2.rectangle(img, box[0], box[1], BLUE, 2)
	for box in GT_boxes:
		cv2.rectangle(img, box[0], box[1], GREEN, 1)
	cv2.imshow(image_name, img)
	cv2.waitKey(0)


# for image in filenames:
# 	draw_boxes(image, images_GT[image], images_YOLO_conf_over_25[image][0])


# image1 = 'resized_93_flip.jpg'
# image2 = 'resized_48.jpg'
# image3 = 'resized_80.jpg'
# image4 = 'resized_106_flip.jpg'
# draw_boxes(image1, images_GT[image1], images_YOLO_conf_over_25[image1][0])
# draw_boxes(image2, images_GT[image2], images_YOLO_conf_over_25[image2][0])
# draw_boxes(image3, images_GT[image3], images_YOLO_conf_over_25[image3][0])
# draw_boxes(image4, images_GT[image4], images_YOLO_conf_over_25[image4][0])