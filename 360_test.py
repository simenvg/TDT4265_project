import cv2
from darknetpy.detector import Detector
import time
import os


class Detected_object():
    def __init__(self, left, right, top, bottom, clss, prob):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.clss = clss
        self.prob = prob
    
    def drawBox(self, img, color):
        cv2.rectangle(img, (self.left,self.top), (self.right,self.bottom), color, 1)
        font = cv2.FONT_HERSHEY_PLAIN 
        cv2.putText(img, self.clss + ' ' + str(round(self.prob,3)), (self.left, self.top - 3),font, 0.8, color, 1)



colors = [(0,255,0), (0,0,255), (255,0,0), (255,255,0), (0,255,255), (255,0,255), (0,125,125), (125,0,125), (125,125,0)]
darknet_path = '/home/simenvg/environments/my_env/darknet'


detector = Detector(darknet_path,
                    darknet_path + '/cfg/coco.data',
                    darknet_path + '/cfg/yolo.cfg',
                    darknet_path + '/yolo.weights')



def detect(img_path):
    result = detector.detect(img_path)
    objects = []
    for elem in result:
        if elem['class'] == 'boat':
            objects.append(Detected_object(elem['left'], elem['right'], elem['top'], elem['bottom'], elem['class'], elem['prob']))
    return objects

def draw_boxes(detected_objects, img, colors):
    for i in range(len(detected_objects)):
        detected_objects[i].drawBox(img, colors[i])



im_width = 1920
im_height = 320

num_subframes = int(1920 / 320) # = 6


start = time.time()
print("Start")

path = '/home/simenvg/environments/my_env/prosjektoppgave/frame'
cap = cv2.VideoCapture('boat360.m4v')
im_count = 0

while(1):
    ret, frame = cap.read()
    if(ret == False):
        break
    
    images = []
    for i in range(num_subframes):
        images.append(frame[0:im_height, i*im_height:(i+1)*im_height])
        cv2.imwrite(os.path.join(path+str(i), 'temp_'+str(im_count)+'.png'),images[i])
        objects = detect(os.path.join(path+str(i), 'temp_'+str(im_count)+'.png'))
        print('WOOOOOW')
        end = time.time()
        print('TIME:  ', end - start)
        if(len(objects) > 0):
            draw_boxes(objects, images[i], colors)
            cv2.imwrite(os.path.join(path+str(i), 'temp_'+str(im_count)+'.png'),images[i])
            # cv2.imshow('frame' + str(i),images[i])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        im_count += 1
                    
    #time.sleep(3)
cap.release()
cv2.destroyAllWindows()
end = time.time()
print('TIME:  ', end - start)


