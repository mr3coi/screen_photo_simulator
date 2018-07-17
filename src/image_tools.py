import numpy as np
import cv2

from scipy.fftpack import dct, idct

def dctII(image: np.array):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

def idctII(array: np.array):
    return idct(idct(array, axis=1, norm='ortho'), axis=0, norm='ortho')

def gamma_correction(img, gamma=2.2):
    invgamma = 1.0/gamma
    table = np.array([(val / 255.0)**invgamma * 255 \
            for val in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def psnr(img1, img2):
    img1 = np.float64(img1); img2 = np.float64(img2)
    for img in (img1, img2):
        if np.amax(img) <= 1:
            img *= 255

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# polar to cartesian
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def visualize_dct(arr: np.array):
    arr = np.log(np.abs(arr))
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return np.floor(arr * 255)

def histogram_equalize(array: np.array, num_bins=1000):
    array = array + np.abs(array.min())
    h = np.histogram(array, bins=num_bins, range=(0, array.max()))
    c = 255 * np.cumsum(h[0]) / np.sum(h[0])

    new_img = np.zeros(array.shape)
    max_val = array.max()
    for index,value in np.ndenumerate(array):
        new_img[index] = c[int((num_bins-1) * value / max_val)]
    return np.floor(new_img)

# Colors (BGR format)
lightgray = (180,180,180)
