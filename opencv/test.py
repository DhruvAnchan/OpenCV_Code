import numpy as np
import cv2
import glob
import os
import plotly.graph_objects as go

# === Load synthetic images and light directions ===
folder = "synthetic_images"
image_files = sorted(glob.glob(os.path.join(folder, "image*.png")))
mask = cv2.imread(os.path.join(folder, "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
light_dirs = np.loadtxt(os.path.join(folder, "light_directions.txt"))

# === Load image stack ===
h, w = mask.shape
num_images = len(image_files)
image_stack = np.zeros((h, w, num_images), dtype=np.float32)

for i, fname in enumerate(image_files):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    image_stack[:, :, i] = img

# === Reshape for solving I = L * N * albedo ===
I = image_stack[mask].T  # shape: (num_images, num_pixels)
L = light_dirs  # shape: (num_images, 3)

# === Solve for g = N * albedo per pixel ===
g, _, _, _ = np.linalg.lstsq(L, I, rcond=None)
normals = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-6)
albedo = np.linalg.norm(g, axis=0)

# === Reconstruct normal map ===
normal_map = np.zeros((h, w, 3), dtype=np.float32)
albedo_map = np.zeros((h, w), dtype=np.float32)
normal_map[mask] = normals.T
albedo_map[mask] = albedo

# === Integrate normals to get depth map ===
fx = -normal_map[:, :, 0] / (normal_map[:, :, 2] + 1e-6)
fy = -normal_map[:, :, 1] / (normal_map[:, :, 2] + 1e-6)

depth = np.zeros_like(fx)
for y in range(1, h):
    depth[y, 0] = depth[y - 1, 0] + fy[y, 0]
for x in range(1, w):
    depth[:, x] = depth[:, x - 1] + fx[:, x]
depth -= np.min(depth)

# === Prepare for 3D rendering ===
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = depth
x = X[mask]
y = Y[mask]
z = Z[mask]
intensity = albedo_map[mask]

# === Surface plot ===
surface = go.Mesh3d(
    x=x, y=y, z=z,
    intensity=intensity,
    colorscale='Viridis',
    showscale=True,
    opacity=1.0,
    alphahull=0
)

# === Surface normals (quiver-like arrows) ===
# Downsample for clarity
step = 10
X_small = X[::step, ::step]
Y_small = Y[::step, ::step]
Z_small = Z[::step, ::step]
N_small = normal_map[::step, ::step, :]

mask_small = mask[::step, ::step]
Xn = X_small[mask_small]
Yn = Y_small[mask_small]
Zn = Z_small[mask_small]
Nx = N_small[:, :, 0][mask_small]
Ny = N_small[:, :, 1][mask_small]
Nz = N_small[:, :, 2][mask_small]

# Scale arrows
arrow_scale = 10
normals_plot = go.Cone(
    x=Xn, y=Yn, z=Zn,
    u=Nx, v=Ny, w=Nz,
    sizemode="scaled",
    sizeref=0.5,
    anchor="tail",
    colorscale='Blues',
    showscale=False,
    name="Normals"
)

# === Plot everything ===
fig = go.Figure(data=[surface, normals_plot])
fig.update_layout(
    title='3D Photometric Stereo Reconstruction with Surface Normals',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Depth',
        aspectmode='data'
    )
)
import plotly.io as pio
pio.renderers.default = 'browser'

fig.show()
