import cv2
import numpy as np

def show_resized_window(title, img, max_width=1280, max_height=720):
    """
    Displays an image resized to fit within max screen dimensions while preserving aspect ratio.
    """
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)

    if scale < 1:  # Resize only if the image is larger than screen limits
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imshow(title, img)

# Load the images
template = cv2.imread('eldenring.jpg', cv2.IMREAD_GRAYSCALE)
scene = cv2.imread('games_onground.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(template, None)
kp2, des2 = sift.detectAndCompute(scene, None)

# Use FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test to filter good matches
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# Minimum number of matches required for finding homography
MIN_MATCH_COUNT = 10

if len(good) >= MIN_MATCH_COUNT:
    # Get keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get bounding box corners of template image
    h, w = template.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Draw detected region in scene
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    cv2.polylines(scene_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Draw matches
    result = cv2.drawMatches(template, kp1, scene_color, kp2, good, None,
                             matchesMask=mask.ravel().tolist(),
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the result resized to fit screen
    show_resized_window('Detected Template in Scene', result)
else:
    print("Not enough matches found - {}/{}".format(len(good), MIN_MATCH_COUNT))

cv2.waitKey(0)
cv2.destroyAllWindows()
