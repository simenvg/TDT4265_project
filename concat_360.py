import cv2
import numpy as np
import os
from PIL import Image
import PIL

path = '/home/simenvg/environments/my_env/prosjektoppgave/frame'

new_path = '/home/simenvg/environments/my_env/prosjektoppgave/images'

im_width = 1920
im_height = 320
num_subframes = int(1920 / 320) # = 6


line = np.zeros((im_height,5,3), np.uint8)


im_count = 0

while (1):
	empty_frame = np.zeros((im_height,1,3), np.uint8)
	list_im = []
	imgs    = []

	for i in range(num_subframes):
		list_im.append(os.path.join(path+str(i), 'temp_'+str(im_count)+'.png'))
		im_count += 1

	imgs    = [ PIL.Image.open(i) for i in list_im ]
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

	imgs_comb = PIL.Image.fromarray( imgs_comb)
	imgs_comb.save(os.path.join(new_path, 'image_'+str(im_count)+'.png' ))
