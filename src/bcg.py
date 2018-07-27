import cv2
import numpy as np

from image_tools import gamma_correction, contrast_brightness

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

    # Image-related
    parser.add_argument("--canvas-dim", type=int, nargs='+', default=1024,
                        help="Dimensions (height, width) of the canvas to use. \
                                Provide a single value to produce a square canvas.")
    parser.add_argument('-g', "--gamma", type=float, default=1,
                        help="Do gamma correction on the given input (default: 1 => no correction)")
    parser.add_argument('-b', "--brightness", type=float, default=0,
                        help="Level of brightness to increase (default: 0 => no correction)")
    parser.add_argument('-c', "--contrast", type=float, default=1,
                        help="Ratio of contrast to apply (default: 1 => no correction)")

    # Others
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed value for 'np.random.seed'.")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    canvas = cv2.imread(os.path.join(args.datapath, args.file), cv2.IMREAD_COLOR)
    original = canvas.copy()
    H, W, _ = canvas.shape

    # ================================== Add operations here =====================================
    '''
    Appropriate settings:
    - g 1.3, c 1.5, b -60
    - g 1.8, c 2, b -110
    '''

    canvas = gamma_correction(canvas, args.gamma)
    canvas = contrast_brightness(canvas, bright=args.brightness, contrast=args.contrast)
    # ===========================================================================================

    # Display result
    cv2.namedWindow("original")
    cv2.moveWindow("original", 40, 60)
    cv2.namedWindow("modified")
    cv2.moveWindow("modified", 40, 60)

    cv2.imshow("original", original)
    cv2.imshow("modified", canvas)

    cv2.waitKey(0)

    # Save output
    if args.save:
        if not os.path.isdir(args.savepath):
            os.makedirs(args.savepath)
        save_name = args.save + '_g{}'.format(args.gamma)
        save_name += '.{}'.format(args.save_format)
        cv2.imwrite(os.path.join(args.savepath, save_name), canvas)

if __name__ == "__main__":
    main()
