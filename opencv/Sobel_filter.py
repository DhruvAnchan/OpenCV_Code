import cv2
import numpy as np
import matplotlib.pyplot as plt

def kernel_mult(image,s):
    x = 1
    y = 2
    if(s == 1):
        kernel = np.array([[x, 0, -x],
                           [y, 0, -y],
                           [x, 0, -x]])
    elif(s == 2):
        kernel = np.array([[x, y, x],
                           [0, 0, 0],
                           [-x, -y, -x]])

    return cv2.filter2D(image, -1, kernel)
#grayscale image
gray_image = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
#Sobel horizontal
image_arr = kernel_mult(gray_image,1)
#sobel vertical
image_arr2 = kernel_mult(gray_image,2)
#sobel combined
image_arr3 = cv2.add(image_arr, image_arr2)

images = [gray_image, image_arr, image_arr2, image_arr3]
titles = ["Original Image", "Sobel Horizontal", "Sobel Vertical", "Sobel Combined"]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()