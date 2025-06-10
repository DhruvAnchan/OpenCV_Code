import numpy as np
import cv2
import matplotlib.pyplot as plt

# === Parameters ===
h, w = 256, 256
x = np.linspace(-1, 1, w)
y = np.linspace(-1, 1, h)
X, Y = np.meshgrid(x, y)

# === Generate Gaussian bump (height map) ===
Z = np.exp(-5 * (X**2 + Y**2))  # smooth hill

# === Compute normals from height map ===
fx, fy = np.gradient(Z)
normals = np.dstack((-fx, -fy, np.ones_like(Z)))
normals /= np.linalg.norm(normals, axis=2, keepdims=True)

# === Define light directions ===
light_dirs = np.array([
    [0, 0, 1],
    [0.5, 0.5, 1],
    [-0.5, 0.5, 1]
], dtype=np.float32)
light_dirs /= np.linalg.norm(light_dirs, axis=1, keepdims=True)

# === Generate images under each lighting condition ===
images = []
for i, L in enumerate(light_dirs):
    # Dot product between normal and light direction
    I = np.clip((normals @ L), 0, 1)  # Lambertian shading
    I_img = (I * 255).astype(np.uint8)
    cv2.imwrite(f'img{i+1}.png', I_img)
    images.append(I_img)

# === Show generated images ===
plt.figure(figsize=(12, 4))
for i, img in enumerate(images):
    plt.subplot(1, 3, i+1)
    plt.title(f'Image {i+1}')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
