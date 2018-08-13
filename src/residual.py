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
    parser.add_argument("--datapath", default='../data/residual_analysis_data',
                        help="Path to the directory containing the source image.")
    parser.add_argument("--file", default='residual_severe_1.png',
                        help="Name of the source image file.")
    parser.add_argument("--savepath", default='../data/output/residual_analysis',
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

    residual = cv2.imread(os.path.join(args.datapath, args.file), cv2.IMREAD_COLOR).transpose((2,0,1))
    H, W, _ = residual.shape
    axis = 0
    edge_len = {0:H,1:W}[axis]

    # ================================== Add operations here =====================================
    for idx, residual_layer in zip('BGR',residual):
        '''
        dct_output = dct(residual_layer, axis=axis, norm='ortho')
        dct_visual = visualize_dct(dct_output)

        modified = dct_output.copy()
        modified[:edge_len//6,:] = 0
        modified_dct_visual = visualize_dct(modified)
        modified = idct(modified, axis=axis, norm='ortho')

        # ===========================================================================================

        # Display result
        cv2.namedWindow("original_{}".format(idx), cv2.WINDOW_NORMAL); cv2.moveWindow("original_{}".format(idx), 40, 60)
        cv2.imshow("original_{}".format(idx), residual_layer)
        cv2.namedWindow("dct_{}".format(idx), cv2.WINDOW_NORMAL); cv2.moveWindow("dct_{}".format(idx), 600, 60)
        cv2.imshow("dct_{}".format(idx), dct_visual)
        cv2.namedWindow("modified_dct_{}".format(idx), cv2.WINDOW_NORMAL); cv2.moveWindow("modified_dct_{}".format(idx), 600, 460)
        cv2.imshow("modified_dct_{}".format(idx), modified_dct_visual)
        cv2.namedWindow("modified_{}".format(idx), cv2.WINDOW_NORMAL); cv2.moveWindow("modified_{}".format(idx), 40, 460)
        cv2.imshow("modified_{}".format(idx), modified)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # Save output
        if args.save:
            if not os.path.isdir(args.savepath):
                os.makedirs(args.savepath)
            save_name += '.{}'.format(args.save_format)
            cv2.imwrite(os.path.join(args.savepath, save_name), residual)
        '''

        img = residual_layer
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
        sobely2 = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=-1)

        print(sobely[:,W//2])
        print(sobely2[:,W//2])
        sum_grad = np.sum(sobely**2,axis=1)
        sum_grad2 = np.sum(sobely2**2,axis=1)

        rot = transforms.Affine2D().rotate_deg(-90)

        plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        '''
        plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        '''
        plt.subplot(2,3,2),plt.imshow(sobely,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,3),plt.plot(sum_grad, transform=rot + plt.gca().transData)
        plt.title('Sobel Y_sum_dist'), plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,5),plt.imshow(sobely2,cmap = 'gray')
        plt.title('Sobel Y2'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,3,6),plt.plot(sum_grad2, transform=rot + plt.gca().transData)
        plt.title('Sobel Y2_sum_dist')
        plt.show()

def visualize_dct(arr: np.array):
    dct_visual = np.log(np.abs(arr)+1)
    dct_visual *= 255/np.max(dct_visual)
    return dct_visual.astype(np.uint8)

if __name__ == "__main__":
    main()
