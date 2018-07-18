import cv2
import numpy as np

from image_tools import *
from moire import linear_wave, dither
from basic_shapes import circles
from module import RecaptureModule

import argparse
import os

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
    parser.add_argument("--save", type=str, default=None,
                        help="Name of the output file storing the results \
                                (not saved if not provided).")
    parser.add_argument("--save-format", type=str, default='jpg',
                        help="File format of the output file (default: JPEG).")

    # Image-related
    parser.add_argument("--canvas-dim", type=int, nargs='+', default=1024,
                        help="Dimensions (height, width) of the canvas to use. \
                                Provide a single value to produce a square canvas.")
    parser.add_argument('-e', "--empty", action='store_true',
                        help="Create a white blank canvas, instead of using an image")
    parser.add_argument('-g', "--gamma", type=float, default=1,
                        help="Do gamma correction on the given input (default: 1 => no correction)")
    parser.add_argument('-t', "--type", type=str, default='fixed',
                        help="Type of pattern to generate.")
    parser.add_argument('-rv', "--recapture-verbose", action='store_true',
                        help="Print the log of progress produced as \
                                RecaptureModule transforms the input image.")

    # Others
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed value for 'np.random.seed'.")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.empty:
        canvas_dim = args.canvas_dim if type(args.canvas_dim) == list else [args.canvas_dim,] * 2
        canvas = np.ones(canvas_dim + [3,], np.uint8) * 255   # blank white image
    else:
        canvas = cv2.imread(os.path.join(args.datapath, args.file), cv2.IMREAD_COLOR)
        original = canvas.copy()
        if args.gamma != 1:
            canvas = gamma_correction(canvas, args.gamma)
    H, W, _ = canvas.shape

    # ================================== Add operations here =====================================
    recap_module = RecaptureModule(v_moire=2, v_type='sg', v_skew=[20, 80], v_cont=10, v_dev=3,
                                   h_moire=2, h_type='f', h_skew=[20, 80], h_cont=10, h_dev=3,
                                   gamma=2.2)
    canvas = recap_module(canvas, verbose=args.recapture_verbose)

    #canvas = dither(canvas,gap=10, skew=50, pattern='rgb', contrast=30, rowwise=True)
    '''
    lineNSkew(canvas, gap=1, skew=50, thick=1, color=(255,255,255))

    circles(canvas, [(H//2,W//2-64),(H//2,W//2+64)], max_rad=H//2,color=lightgray)
    radialShape(canvas, (H//2-128,W//2-128), 512, 60, thick=2, color=lightgray)
    radialShape(canvas, (H//2+128,W//2+128), 512, 60, thick=2, color=lightgray)
    '''
    # ===========================================================================================

    # Display result
    cv2.imshow("modified", canvas)
    if not args.empty:
        cv2.imshow("original", original)
        psnr_val = psnr(canvas,original)
        print("PSNR value: %.4f db" % psnr_val)
    cv2.waitKey(0)

    # Save output
    if not args.empty and args.save:
        if not os.path.isdir(args.savepath):
            os.makedirs(args.savepath)
        save_name = args.save + '_g{}'.format(args.gamma)
        save_name += '_{:.4f}'.format(psnr_val)
        save_name += '.{}'.format(args.save_format)
        cv2.imwrite(os.path.join(args.savepath, save_name), canvas)

if __name__ == "__main__":
    main()
