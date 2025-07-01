import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# === Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,              # 3D model
    enable_segmentation=False,
    smooth_landmarks=True            # Important for reducing jitter
)
mp_drawing = mp.solutions.drawing_utils

def get_3d_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_angles_and_frame(frame):
    h, w, _ = frame.shape
    center_crop = frame[:, w//4 : 3*w//4]
    image = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark
    points = np.array([
        [l.x * center_crop.shape[1], l.y * center_crop.shape[0], l.z * center_crop.shape[1]]
        for l in landmarks
    ])

    try:
        a = mp_pose.PoseLandmark
        angles = []

        # 3D Joint Angles
        def ang(i, j, k): return get_3d_angle(points[a[i].value], points[a[j].value], points[a[k].value])
        angles += [
            ang('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            ang('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            ang('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            ang('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            ang('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
            ang('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
            ang('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
            ang('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
            ang('LEFT_SHOULDER', 'LEFT_HIP', 'RIGHT_SHOULDER'),
            ang('RIGHT_SHOULDER', 'RIGHT_HIP', 'LEFT_SHOULDER'),
            ang('LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_HIP'),
            ang('RIGHT_HIP', 'RIGHT_SHOULDER', 'LEFT_HIP'),
            ang('LEFT_WRIST', 'LEFT_ELBOW', 'LEFT_SHOULDER'),
            ang('RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER'),
            ang('LEFT_KNEE', 'LEFT_HIP', 'LEFT_SHOULDER'),
            ang('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_SHOULDER')
        ]

    except:
        return None, None

    mp_drawing.draw_landmarks(center_crop, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    frame[:, w//4:3*w//4] = center_crop
    return np.array(angles), frame 

def extract_all_angles_and_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    angle_vectors, frames = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        angles, annotated = extract_angles_and_frame(frame)
        if angles is not None:
            angle_vectors.append(angles)
            frames.append(annotated)

    cap.release()
    return angle_vectors, frames

def compare_with_dtw(vecs1, vecs2):
    distance, path = fastdtw(vecs1, vecs2, dist=euclidean)
    print("\n" + "=" * 60)
    print(f"DTW Total Distance: {distance:.2f}")
    return path

def get_cosine_verdict(avg_cosine):
    if avg_cosine > 0.95:
        return "very_similar"
    elif avg_cosine > 0.85:
        return "moderately_similar"
    else:
        return "dissimilar"

def get_euclidean_verdict(avg_euclidean):
    if avg_euclidean < 0.5:
        return "very_similar"
    elif avg_euclidean < 0.8:
        return "moderately_similar"
    else:
        return "dissimilar"

def final_verdict(e_v, c_v):
    if c_v == "very_similar" or e_v == "very_similar":
        return "VERY SIMILAR"
    elif c_v == "moderately_similar" and e_v == "moderately_similar":
        return "MODERATELY SIMILAR"
    else:
        return "DISSIMILAR"


def generate_dtw_video(vecs1, frames1, vecs2, frames2, path, out_path="dtw_output_3d.mp4"):
    height, width = 480, 640
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width*2, height))

    euclideans, similarities = [], []

    for i1, i2 in path:
        if i1 >= len(frames1) or i2 >= len(frames2):
            continue

        v1 = vecs1[i1] / 180.0
        v2 = vecs2[i2] / 180.0

        eu = np.linalg.norm(v1 - v2)
        co = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)

        euclideans.append(eu)
        similarities.append(co)

        f1 = cv2.resize(frames1[i1], (width, height))
        f2 = cv2.resize(frames2[i2], (width, height))
        combined = np.hstack((f1, f2))

        cv2.putText(combined, f"Euclidean: {eu:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, f"Cosine: {co:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out.write(combined)

    out.release()
    print(f"DTW video saved: {out_path}")

    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(euclideans); plt.grid(); plt.title("Euclidean Distance Over Time")
    plt.subplot(1, 2, 2)
    plt.plot(similarities); plt.grid(); plt.title("Cosine Similarity Over Time")
    plt.tight_layout(); plt.show()
    
    avg_eu, avg_co = np.mean(euclideans), np.mean(similarities)
    MAX_EUCLIDEAN = np.sqrt(len(vecs1[0]))
    score_eu = max(0, 100 - (avg_eu / MAX_EUCLIDEAN) * 100)
    score_co = avg_co * 100
    total_score = score_eu * 0.4 + score_co * 0.6

    e_v, c_v = get_euclidean_verdict(avg_eu), get_cosine_verdict(avg_co)
    verdict = final_verdict(e_v, c_v)
    result = dict();
    result['Average Euclidean'] = avg_eu
    result['Average Cosine'] = avg_co
    result['Euclidean Score'] = score_eu
    result['Cosine Score'] = score_co
    result["Total Score"] = total_score
    result["Verdict"] = verdict
    return result