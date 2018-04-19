import cv2
import os
import numpy as np
import copy
import pickle

folder_path = '/home/simenvg/environments/my_env/datasyn/project/JPEGImages'


# images2 = pickle.load(open(os.path.join(folder_path,'boxes.txt'), "rb"))

# print('HEHREIWAHFSJDFAJSDOF')
# for keys,values in images2.items():
#     print(keys)
#     print(values)

# print('GADHFgdjkfhgiksdhruioøghaeiorgvjndklg')




images = {}


def draw_rectangle(event, x, y, flags, params):
    global start_point, end_point
    if event == cv2.EVENT_LBUTTONDOWN:
         print('Start Mouse Position: '+str(x)+', '+str(y))
         start_point = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: '+str(x)+', '+str(y))
        end_point = (x, y)
        cv2.rectangle(img, start_point, end_point, (255,0,0), 1)


for filename in os.listdir(folder_path):
    if filename == 'ground_truth.txt' or filename == 'YOLO.txt':
        continue
    img = cv2.imread(os.path.join(folder_path,filename))
    #img = np.zeros((512,512,3), np.uint8)
    img_true = copy.copy(img)
    img_backup = copy.copy(img)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_rectangle)


    boxes = []

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) #& 0xFF
        if k == 27:
            break
        elif k== -1:
            continue
        elif k == 8: #backspace
            img = copy.copy(img_true)
        elif k == 13: # return
            img_true = copy.copy(img)
            boxes.append([start_point, end_point])
            print('HEYYY  ', boxes)
        elif k == 80: # home
            img = copy.copy(img_backup)
            img_true = copy.copy(img)
        else:
            print(k)
    
    images[filename] = boxes
    cv2.destroyAllWindows()


pickle.dump(images, open(os.path.join(folder_path,'ground_truth.txt'), "wb"))


# for keys,values in images.items():
#     print(keys)
#     print(values)