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

def gamma_correction01(img, gamma=2.2):
    invgamma = 1.0/gamma
    return img**invgamma

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

def to01float(image: np.array):
    return (image.astype(float) / 255).clip(0,1)

def to255uint8(image: np.array):
    return (image * 255).clip(0,255).astype(np.uint8)

def contrast_brightness(image: np.array, bright=0.0, contrast=1.0):
    out = (image.astype(float) - 127) * contrast + 127 + bright
    return out.clip(0,255).astype(np.uint8)

def contrast_brightness01(image: np.array, bright=1.0, contrast=1.0):
    out = (image - 0.5) * contrast + 0.5
    out *= bright
    return out.clip(0,1)

def rgb2hsv(rgb, input11=False):
    def convert11to01(val):
        return (val+1)/2

    if input11:
        rgb = [convert11to01(val) for val in rgb]

    if type(rgb[0]) == float:
        r, g, b = [255 * val for val in rgb]
    else:
        r, g, b = [float(val) for val in rgb]

    min_val = min(r, g, b)
    max_val = max(r, g, b)
    v = max_val

    chroma = max_val - min_val
    if v != 0:
        s = chroma / v
    else:
        s = 0
        h = -1      # unknown
        return h, s, v

    if r == max_val:
        h = (g-b) / chroma
    elif g == max_val:
        h = (b-r) / chroma + 2
    else:
        h = (r-g) / chroma + 4
    h *= 60
    h = h + 360 if h < 0 else h

    return h, s, v

def hsv2rgb(hsv, output='255'):
    def convert01to11(val):
        return val*2-1

    h, s, v = hsv

    if h == -1:
        return 0,0,0

    rgb = np.zeros(3)
    offsets = [0,2,4]

    # Determine max and set value
    if h >= 60 and h < 180:
        max_idx = 1         # G is max
    elif h >= 180 and h < 300:
        max_idx = 2         # B is max
    else:
        max_idx = 0         # R is max
    rgb[max_idx] = v

    chroma = s * v
    min_val = v - chroma

    diff = (h / 60 - offsets[max_idx]) * chroma
    if diff > 0:
        min_idx = (max_idx+1) % 3
        mid_idx = (max_idx+2) % 3
    else:
        min_idx = (max_idx+2) % 3
        mid_idx = (max_idx+1) % 3
    rgb[min_idx] = min_val
    rgb[mid_idx] = min_val + abs(diff)

    if output == '01':
        rgb = [val / 255 for val in rgb]
    elif output == '11':
        rgb = [convert01to11(val / 255) for val in rgb]
    elif output == '255':
        rgb = np.uint8(rgb)
    else:
        raise ValueError("Output type must be one of among '01', '11', '255'.")
    return rgb

def img_convert(img: np.array, func):
    return np.array([func(pxl) for pxl in img.reshape(-1,3)]).reshape(img.shape)

def adjust_hue(img: np.array, hue: float):
    '''
    :param hue: (degree to rotate) / 360
    :type hue: float (0-1)
    '''
    # For convenience of multiplication
    H,W,C = img.shape
    img = img.reshape((H,W,C,1))

    # Convert img to YIQ format
    T_yiq = np.array([[.299,.587,.114],[.596,-.274,-.321],[.211,-.523,.311]])
    assert C==3, "The number of channels in the given image is {} instead of 3.".format(C)
    img_yiq = np.matmul(T_yiq, img)

    # Apply hue change
    hue *= 2*np.pi
    U = np.cos(hue); V = np.sin(hue)
    T_hue = np.array([[1,0,0],[0,U,-V],[0,V,U]])
    img_hue = np.matmul(T_hue, img_yiq)

    # Convert back to RGB format
    T_yiq_inv = np.array([[1,.956,.621],[1,-.272,-.647],[1,-1.107,1.705]])
    out = np.matmul(T_yiq_inv, img_hue)

    # Revert back to original dimensions
    out = out.reshape((H,W,C))

    return out.clip(0,1)

def hue_noise(img: np.array, noise=None, max_size=None):
    assert noise or max_size, "Provide either the noise values or a max size limit."
    if noise is None:
        noise = np.array([np.random.randint(max_size) / 255 for _ in range(3)])
    else:
        assert type(noise) == list and len(noise) == 3
        if np.any([type(val) != float for val in noise]):
            noise = [val / 255 for val in noise]
        noise = np.array(noise)
    out = img + noise.reshape((1,1,-1))
    return out.clip(0,1)
    
# Colors (BGR format)
lightgray = (180,180,180)
