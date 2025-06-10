import cv2
import numpy as np
import matplotlib.pyplot as plt


def zoom_image(image, scale, interpolation):
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=interpolation)


def logarithmic_transform(image, c=1):
    image = np.float32(image) / 255.0
    log_transformed = c * np.log(1 + image)
    log_transformed = np.uint8(255 * log_transformed / np.max(log_transformed))
    return log_transformed


def power_law_transform(image, gamma=0.8):
    image = np.float32(image) / 255.0
    power_law = np.uint8(255 * (image ** gamma))
    return power_law


def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched


def histogram_equalisation(image):
    return cv2.equalizeHist(image)


def histogram_matching(image, reference):
    bins = 256
    hist, bins = np.histogram(image.flatten(), bins, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]

    ref_hist, bins = np.histogram(reference.flatten(), bins, [0, 256])
    ref_cdf = ref_hist.cumsum()
    ref_cdf_normalized = ref_cdf * 255 / ref_cdf[-1]

    lookup_table = np.interp(cdf_normalized, ref_cdf_normalized, np.arange(256))
    matched = np.interp(image.flatten(), np.arange(256), lookup_table)
    return matched.reshape(image.shape).astype(np.uint8)


def smoothing_filter(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


def sharpening_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# Load Image
gray_image = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)

# Perform operations
zoom_nn = zoom_image(gray_image, 2, cv2.INTER_NEAREST)
zoom_bilinear = zoom_image(gray_image, 2, cv2.INTER_LINEAR)
zoom_bicubic = zoom_image(gray_image, 2, cv2.INTER_CUBIC)

log_transformed = logarithmic_transform(gray_image)
power_law_transformed = power_law_transform(gray_image)
contrast_stretched = contrast_stretching(gray_image)
hist_eq = histogram_equalisation(gray_image)

# Create reference image for histogram matching
y = np.arange(256)
reference = np.uint8(0.5 * y)
matched_image = histogram_matching(gray_image, reference)

smooth_linear = smoothing_filter(gray_image)
smooth_order_stat = median_filter(gray_image)
sharpened = sharpening_filter(gray_image)

# Display results
images = [gray_image, zoom_nn, zoom_bilinear, zoom_bicubic, log_transformed,
          power_law_transformed, contrast_stretched, hist_eq, matched_image,
          smooth_linear, smooth_order_stat, sharpened]
titles = ['Original', 'Zoom Nearest Neighbour', 'Zoom Bilinear', 'Zoom Bicubic', 'Log Transform (c = 1)',
          'Power Law (gamma = 0.8)', 'Contrast Stretching', 'Histogram Equalization', 'Histogram Matching (y = 0.5x)',
          'Smooth Linear', 'Smooth Order Stat', 'Sharpened']

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
