import cv2
import os

# --- Parameters ---
image_folder = 'stitching/'
screen_width = 1280  # Change based on your screen resolution
screen_height = 720  # Change based on your screen resolution

# --- Load images ---
image_files = sorted([
    os.path.join(image_folder, f) for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
images = [cv2.imread(file) for file in image_files]

# --- Error check ---
if any(img is None for img in images):
    print("One or more images couldn't be loaded. Check the file paths.")
    exit()

# --- Stitch images ---
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    print("Stitching successful!")

    # Resize stitched image to fit screen
    h, w = stitched.shape[:2]
    scale_w = screen_width / w
    scale_h = screen_height / h
    scale = min(scale_w, scale_h)
    resized = cv2.resize(stitched, (int(w * scale), int(h * scale)))

    # Show and save output
    cv2.imshow("Panorama (Fitted)", resized)
    cv2.imwrite("panorama_output.jpg", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Stitching failed with status code:", status)
