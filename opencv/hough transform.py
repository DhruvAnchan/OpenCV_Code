import cv2
import numpy as np

# Load the image
img = cv2.imread('coins.jpg')
output = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise to avoid false circle detection
gray_blurred = cv2.medianBlur(gray, 5)

# Apply Hough Circle Transform
circles = cv2.HoughCircles(gray_blurred,
                           cv2.HOUGH_GRADIENT,
                           dp=1.2,
                           minDist=30,
                           param1=50,
                           param2=50,
                           minRadius=30,
                           maxRadius=75)

# If circles are detected
if circles is not None:
    circles = np.uint16(np.around(circles))  # round values
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

# Show the result
cv2.imshow('Detected Circles', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
