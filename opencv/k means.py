import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('water_coins.jpg')
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
images = []

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
for x in range(9):
    K = x+1
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    images.append(res2)

num_images = len(images)
cols = 3  # Number of columns
rows = (num_images + cols - 1) // cols
plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i], cmap='gray')
    text = "iteration "+ str(i)
    plt.title(text)
    plt.axis('off')
plt.tight_layout()
plt.show()