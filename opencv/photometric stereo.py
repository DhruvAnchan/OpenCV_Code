import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load Images ===
image_files = ['007.png', '008.png', '009.png']
images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in image_files]
images = np.stack(images, axis=-1)  # Shape: (H, W, N)

# === 2. Define Light Directions ===
# These must match the number of images
light_dirs = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1]
], dtype=np.float32)

# Normalize light directions
light_dirs = light_dirs / np.linalg.norm(light_dirs, axis=1, keepdims=True)

# === 3. Solve for Surface Normals and Albedo ===
h, w, n = images.shape
I = images.reshape(-1, n).T  # Shape: (N, H*W)

# Least squares solution: I = L * G  -->  G = (L^T L)^-1 L^T I
L = light_dirs
G = np.linalg.lstsq(L, I, rcond=None)[0]  # Shape: (3, H*W)

# Extract albedo and normals
albedo = np.linalg.norm(G, axis=0)
normals = G / (albedo + 1e-5)

# Reshape to image format
albedo_img = albedo.reshape(h, w)
normals_img = normals.T.reshape(h, w, 3)

# === 4. Integrate Normals to Get Height Map ===
def integrate_normals(normals):
    fx = normals[:, :, 0] / (normals[:, :, 2] + 1e-5)
    fy = normals[:, :, 1] / (normals[:, :, 2] + 1e-5)

    height_map = np.zeros_like(fx)
    for y in range(1, height_map.shape[0]):
        height_map[y, 0] = height_map[y-1, 0] + fy[y, 0]
    for x in range(1, height_map.shape[1]):
        height_map[:, x] = height_map[:, x-1] + fx[:, x]

    return height_map

height_map = integrate_normals(normals_img)

# === 5. Display Results ===
fig = plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title('Albedo')
plt.imshow(albedo_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Normal Map')
normal_vis = (normals_img + 1) / 2  # normalize to [0,1] for display
plt.imshow(normal_vis)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Height Map')
plt.imshow(height_map, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
