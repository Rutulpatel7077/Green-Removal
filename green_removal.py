import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.io as sio
from skimage.io import imread, imsave

#input image
image = imread('download.jpg')


# Define our color selection boundaries in RGB values
lower_green = np.array([41,252,46]) 
upper_green = np.array([41,252,46])

# Define the masked area
mask = cv2.inRange(image, lower_green, upper_green)

# Mask the image to let the car show through
masked_image = np.copy(image)

masked_image[mask != 0] = [0, 0, 0]


# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('H channel')
ax1.imshow(h, cmap='gray')
ax2.set_title('S channel')
ax2.imshow(s, cmap='gray')
ax3.set_title('V channel')
ax3.imshow(v, cmap='gray')


#Settting colour limit
lower_green_hue = np.array([52]) 
upper_green_hue = np.array([66])

# Define the masked area
hue_mask = cv2.inRange(h, lower_green_hue, upper_green_hue)

hue_masked_image = np.copy(h)

# Convert image to monotone image (255 = car, 0 = background)
hue_masked_image[hue_mask != 0] = [0]
hue_masked_image[hue_mask == 0] = [255]


# Mask the image to show real object
masked_image = np.copy(image)
masked_image[hue_masked_image == 0] = [0, 0, 0]

#Write Image
imsave('1.jpg', masked_image)


