import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# === 1. Load Haar cascade for face detection ===
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === 2. Helper: Resize and flatten images for SVM ===
def extract_features(img):
    return cv2.resize(img, (64, 64)).flatten()

# === 3. Create a simple dataset of face and non-face examples ===
def load_data():
    face_images = []
    non_face_images = []

    # Load example face image
    face_img = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)
    if face_img is None:
        print("Image 'test.jpg' not found. Please place it in the same directory.")
        exit()

    faces = haar_cascade.detectMultiScale(face_img, 1.1, 4)
    for (x, y, w, h) in faces:
        roi = face_img[y:y+h, x:x+w]
        face_images.append(extract_features(roi))

    # Generate random noise as non-face examples
    for _ in range(len(face_images)):
        non_face = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        non_face_images.append(extract_features(non_face))

    # Labels: 1 = Face, 0 = Not face
    X = np.array(face_images + non_face_images)
    y = np.array([1] * len(face_images) + [0] * len(non_face_images))
    return train_test_split(X, y, test_size=0.3, random_state=42)

# === 4. Train SVM ===
X_train, X_test, y_train, y_test = load_data()
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
print("✅ SVM trained.\n")
print("📊 Classification Report:\n")
print(classification_report(y_test, svm.predict(X_test)))

# === 5. Detect faces in a new image and classify ===
test_img = cv2.imread('rose.jpg')
if test_img is None:
    print("Error loading test image.")
    exit()

gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
detected = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in detected:
    roi = gray[y:y+h, x:x+w]
    roi_features = extract_features(roi).reshape(1, -1)
    pred = svm.predict(roi_features)
    label = "Face" if pred[0] == 1 else "Not Face"
    color = (0, 255, 0) if label == "Face" else (0, 0, 255)
    cv2.rectangle(test_img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(test_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# === 6. Show and/or save result ===
output_path = "svm_result.jpg"
cv2.imwrite(output_path, test_img)
cv2.imshow("SVM Face Detection", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"💾 Saved result to {output_path}")



