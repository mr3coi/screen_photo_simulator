import cv2
import numpy as np
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt, transforms
from scipy import ndimage

import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument("--datapath", default='../data',
                        help="Path to the directory containing the source image.")
    parser.add_argument("--file", default='original.png',
                        help="Name of the source image file.")
    parser.add_argument("--savepath", default='../data/output',
                        help="Path to the output storage directory \
                                (automatically generated if not yet there).")
    parser.add_argument("--save", type=str, default=None,
                        help="Name of the output file storing the results \
                                (not saved if not provided).")
    parser.add_argument("--save-format", type=str, default='jpg',
                        help="File format of the output file (default: JPEG).")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    original = cv2.imread(os.path.join(args.datapath, args.file), cv2.IMREAD_COLOR).transpose((2,0,1)) / 255.0
    H, W, _ = original.shape
    axis = 0
    edge_len = {0:H,1:W}[axis]

    threshold = 0.3

    # ================================== Add operations here =====================================
    lap_filtered_img = original.copy()
    sobel_filtered_img = original.copy()

    for idx, original_layer, sobel_layer, lap_layer in zip('BGR',original,sobel_filtered_img,lap_filtered_img):
        img = original_layer
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

        candidates = img < threshold

        sobel_sqsum = np.sqrt(sobelx**2 + sobely**2)

        plt.subplot(2,3,1),plt.imshow(img, cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(2,3,2),plt.imshow(np.sqrt(laplacian**2), cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

        '''
        plt.subplot(2,3,3),plt.imshow(np.sqrt(sobelx**2), cmap = 'gray')
        plt.title('Sq. Sobel X'), plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,4),plt.imshow(np.sqrt(sobely**2), cmap = 'gray')
        plt.title('Sq. Sobel Y'), plt.xticks([]), plt.yticks([])
        '''

        '''
        rot = transforms.Affine2D().rotate_deg(-90)

        sum_grad = np.sum(sobely**2,axis=1)
        plt.subplot(2,3,3),plt.plot(sum_grad, transform=rot + plt.gca().transData)
        plt.title('Sobel Y_sum_dist'), plt.xticks([]), plt.yticks([])

        sum_grad2 = np.sum(sobely2**2,axis=1)
        plt.subplot(2,3,6),plt.plot(sum_grad2, transform=rot + plt.gca().transData)
        plt.title('Sobel Y2_sum_dist')
        '''

        '''
        sobely2 = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=-1)
        plt.subplot(2,3,5),plt.imshow(sobely2, cmap = 'gray')
        plt.title('Sobel Y2'), plt.xticks([]), plt.yticks([])
        '''

        plt.subplot(2,3,3),plt.imshow(sobel_sqsum, cmap = 'gray')
        plt.title('Sobel sq sum'), plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,4),plt.imshow(candidates, cmap = 'gray')
        plt.title('Dark areas'), plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,5),plt.imshow(candidates * np.sqrt(laplacian**2), cmap = 'gray')
        plt.title('Filtered Laplacian'), plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,6),plt.imshow(candidates * sobel_sqsum, cmap = 'gray')
        plt.title('Filtered Sobel sq sum'), plt.xticks([]), plt.yticks([])

        plt.show()

        sobel_layer *= (candidates * sobel_sqsum).astype('float64')
        lap_layer *= (candidates * np.sqrt(laplacian**2)).astype('float64')

    bgr2rgb = [2,1,0]
    original = original[bgr2rgb].transpose((1,2,0))
    sobel_filtered_img = sobel_filtered_img[bgr2rgb].transpose((1,2,0))
    lap_filtered_img = lap_filtered_img[bgr2rgb].transpose((1,2,0))

    print('drawing original...')
    plt.subplot(1,3,1), plt.imshow(original)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    print('drawing filtered sobel...')
    plt.subplot(1,3,2), plt.imshow(sobel_filtered_img)
    plt.title('Filtered Sobel'), plt.xticks([]), plt.yticks([])
    print('drawing filtered laplacian...')
    plt.subplot(1,3,3), plt.imshow(sobel_filtered_img)
    plt.title('Filtered Laplacian'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
