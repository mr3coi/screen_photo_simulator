import numpy as np
from scipy.fftpack import fft2
import cv2

import argparse
import os

from image_tools import psnr, dctII, idctII

from scipy.fftpack import dct, idct

def dctII(image: np.array):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

def idctII(array: np.array):
    return idct(idct(array, axis=1, norm='ortho'), axis=0, norm='ortho')

def get_parser():
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument("--datapath", default='../data/sample_images',
                        help="Path to the directory containing the source image.")
    parser.add_argument("--file", default='med_1.jpg',
                        help="Name of the source image file.")
    parser.add_argument("--savepath", default='../data/output',
                        help="Path to the output storage directory \
                                (automatically generated if not yet there).")
    parser.add_argument("--save", type=str, default="linear_fixed_weak_25.0247.jpg",
                        help="Name of the output file storing the results \
                                (not saved if not provided).")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    original = cv2.imread(os.path.join(args.datapath, args.file), cv2.IMREAD_COLOR)
    transformed = cv2.imread(os.path.join(args.savepath, args.save), cv2.IMREAD_COLOR)

    '''
    cv2.imshow("original", original)
    cv2.imshow("transformed", transformed)
    cv2.waitKey(0)
    '''

    '''
    residual = (transformed - original).transpose((2,0,1))
    for channel, layer in zip('RGB', residual):
        dct_output = dctII(layer)
        print(dct_output)
        cv2.imshow(channel, histogram_equalize(dct_output))
    cv2.waitKey(0)
    '''

    added = dct_add_noise(original)
    diff = np.abs(added-original)
    print("sum_diff:", np.sum(diff,axis=(0,1)))
    print("PSNR value:", psnr(original,added))
    names = ['original', 'added', 'diff', 'R_diff', 'G_diff', 'B_diff']
    items =[original, added, diff, diff[:,:,0], diff[:,:,1], diff[:,:,2]]
    for name, img in zip(names, items):
        cv2.imshow(name, img)
    cv2.waitKey(0)

def histogram_equalize(array: np.array, num_bins=1000):
    array = array + np.abs(array.min())
    h = np.histogram(array, bins=num_bins, range=(0, array.max()))
    c = 255 * np.cumsum(h[0]) / np.sum(h[0])

    new_img = np.zeros(array.shape)
    max_val = array.max()
    for index,value in np.ndenumerate(array):
        new_img[index] = c[int((num_bins-1) * value / max_val)]
    return np.floor(new_img)

def dct_add_noise(image):
    def generate_noise(image):
        '''
        Generate 3-D noise (list of 3 items)
        :param image: input image (CHW)
        '''
        noises = []
        for channel in range(len('rgb')):
            noise = np.zeros(image[0].shape)
            #if channel == 2:
            noise[0,200] = image[channel][0][200] * 20
            '''
                noise[1,idx] = amplitude
                noise[0,idx+1] = amplitude
                noise[1,idx+1] = amplitude
            '''
            noises.append(noise)
        print(np.sum(np.array(noises)))
        return noises

    image_dct = [dctII(layer) for layer in image.transpose((2,0,1))]    # HWC -> CHW
    noise = generate_noise(image_dct)
    out = np.array([idctII(layer + noise_layer) for layer, noise_layer in zip(image_dct, noise)])
    print("overflow count:", np.sum(out > 255))
    return np.round(out).transpose((1,2,0)).astype(np.uint8)    # CHW -> HWC

if __name__ == "__main__":
    main()
