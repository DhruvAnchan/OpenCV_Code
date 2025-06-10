import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 1. Load images taken under different lighting ===
image_files = [
    "006.png",
    "007.png",
    "008.png",
    "009.png"
]
images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32) for img in image_files]
h, w = images[0].shape

# === 2. Define corresponding light directions (Nx3) ===
light_dirs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1]
], dtype=np.float32)
light_dirs = light_dirs / np.linalg.norm(light_dirs, axis=1, keepdims=True)  # Normalize

# === 3. Stack image intensities into a 2D matrix (H*W x N) ===
num_images = len(images)
I = np.stack(images, axis=-1).reshape(-1, num_images)

# === 4. Solve for surface normals and albedo using least squares ===
albedo = np.zeros((h * w), dtype=np.float32)
normals = np.zeros((h * w, 3), dtype=np.float32)

for i in range(h * w):
    i_values = I[i]
    g, _, _, _ = np.linalg.lstsq(light_dirs, i_values, rcond=None)
    albedo[i] = np.linalg.norm(g)
    if albedo[i] > 1e-5:
        normals[i] = g / albedo[i]

# === 5. Reshape for image form ===
albedo_img = albedo.reshape((h, w))
normals_img = normals.reshape((h, w, 3))

# === 6. Compute depth using Fourier integration ===
def integrate_normals(normals):
    h, w, _ = normals.shape
    fx = np.zeros((h, w))
    fy = np.zeros((h, w))

    valid = normals[:, :, 2] != 0
    fx[valid] = -normals[:, :, 0][valid] / normals[:, :, 2][valid]
    fy[valid] = -normals[:, :, 1][valid] / normals[:, :, 2][valid]

    FX = np.fft.fft2(fx)
    FY = np.fft.fft2(fy)

    u = np.fft.fftfreq(w)
    v = np.fft.fftfreq(h)
    U, V = np.meshgrid(u, v)

    denom = (2j * np.pi * U)**2 + (2j * np.pi * V)**2
    denom[0, 0] = 1  # avoid divide-by-zero

    Fz = (-1j * 2 * np.pi * U * FX - 1j * 2 * np.pi * V * FY) / denom
    Fz[0, 0] = 0  # remove DC component

    depth = np.real(np.fft.ifft2(Fz))
    return depth

depth_map = integrate_normals(normals_img)

# === 7. Display Albedo and Normals ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Albedo")
plt.imshow(albedo_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Normals (RGB)")
norm_rgb = (normals_img + 1) / 2  # scale from [-1,1] to [0,1]
plt.imshow(norm_rgb)
plt.axis('off')
plt.show()

# === 8. Plot Depth Map as 3D Surface ===
X, Y = np.meshgrid(np.arange(w), np.arange(h))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, depth_map, cmap='viridis', linewidth=0, antialiased=False)
ax.set_title("Reconstructed Depth Map")
plt.show()
