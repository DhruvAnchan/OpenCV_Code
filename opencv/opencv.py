import cv2
import numpy as np

# Step 1: Load the image
image = cv2.imread('vase.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Step 4: Dilate to close small gaps in edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed_edges = cv2.dilate(edges, kernel, iterations=1)

# Step 5: Find contours from the closed edges
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Create a blank canvas to fill
filled = np.zeros_like(edges)

# Step 7: Fill all detected contours
cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

# Optional: Create color version to overlay on original image
filled_color = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(image, 0.6, filled_color, 0.4, 0)

# Display everything
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Closed Edges (Dilated)", closed_edges)
cv2.imshow("Filled Regions from Edges", filled)
cv2.imshow("Overlay on Original", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
