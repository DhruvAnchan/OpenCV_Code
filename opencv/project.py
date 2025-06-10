import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_images(images, titles):
    """Display multiple images in a single window."""
    num_images = len(images)
    cols = 3  # Number of columns
    rows = (num_images + cols - 1) // cols  # Calculate number of rows needed

    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# Load the image
image_path = 'seg.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
for i in range(1):
    image = cv2.GaussianBlur(image, (3,3), 0)
# Boundary Extraction
edges = cv2.Canny(image, 100, 200)

# Hole Filling
inverted_image = cv2.bitwise_not(image)
_, thresh = cv2.threshold(inverted_image, 127, 255, cv2.THRESH_BINARY)
filled_holes = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

# Extraction of Connected Components
num_labels, labels_im = cv2.connectedComponents(filled_holes)

# Convex Hull
contours, _ = cv2.findContours(filled_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull = [cv2.convexHull(c) for c in contours]
convex_hull_image = np.zeros_like(edges)
conv = cv2.drawContours(convex_hull_image, hull, -1, (255), thickness=cv2.FILLED)

# Erosion
kernel = np.ones((13,13), np.uint8)
eroded_image = cv2.erode(filled_holes, kernel, iterations=1)

# Dilation
dilated_image = cv2.dilate(filled_holes, kernel, iterations=1)

# Closing
closed_image = cv2.morphologyEx(filled_holes, cv2.MORPH_CLOSE, kernel)

# Opening
opened_image = cv2.morphologyEx(filled_holes, cv2.MORPH_OPEN, kernel)

# Output
images = [image, edges,  filled_holes, labels_im, convex_hull_image,
          eroded_image, dilated_image, closed_image, opened_image]
titles = ['Original Image', 'Boundary Extraction', 'Hole Filling',
          'Connected Components', 'Convex Hull', 'Erosion',
          'Dilation', 'Closing', 'Opening']

display_images(images, titles)