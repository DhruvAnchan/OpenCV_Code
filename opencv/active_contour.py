import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.segmentation import active_contour

# Load image
img = cv2.imread('coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0.1)

# Normalize image for skimage
gray_norm = gray / 255.0

# Create initial snake (a circle around the object)
s = np.linspace(0, 2*np.pi, 4000)
r = 85 +  70*np.sin(s)  # y coordinates
c = 100+  70*np.cos(s)  # x coordinates
init = np.array([r, c]).T

# Apply active contour (snake) WITHOUT 'coordinates' arg
snake = active_contour(
    gray_norm,
    init,
    alpha=0.05,
    beta=40,
    gamma=0.02
)

# Plot the result
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gray, cmap='gray')
ax.plot(init[:, 1], init[:, 0], '--r', label='Initial contour')
ax.plot(snake[:, 1], snake[:, 0], '-b', label='Final contour')
ax.legend()
plt.show()
