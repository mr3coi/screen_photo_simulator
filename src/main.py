import cv2
import numpy as np

from image_tools import *

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
    canvas = linear_wave(canvas,contrast=10,skew=80)
    canvas = linear_wave(canvas,contrast=10,skew=20)
    canvas = linear_wave(canvas,contrast=10,skew=20,rowwise=False)
    canvas = linear_wave(canvas,contrast=10,skew=80,rowwise=False)

    '''
    dither(canvas,gap=10, skew=50)
    lineNSkew(canvas, gap=1, skew=50, thick=1, color=(255,255,255))

    circles(canvas, [(H//2,W//2-64),(H//2,W//2+64)], max_rad=H//2,color=lightgray)
    radialShape(canvas, (H//2-128,W//2-128), 512, 60, thick=2, color=lightgray)
    radialShape(canvas, (H//2+128,W//2+128), 512, 60, thick=2, color=lightgray)
    '''
    # ===========================================================================================

    # Display result
    cv2.imshow("modified", canvas)
    '''
    cv2.imshow("source-med diff", np.abs(original-med))
    cv2.imshow("intermediate", med)
    cv2.imshow("med-canvas diff", np.abs(canvas-med))
    cv2.imshow("source-canvas diff", np.abs(original-canvas))
    '''
    if not args.empty:
        cv2.imshow("original", original)
        psnr_val = psnr(canvas,original)
        print("PSNR value: %.4f db" % psnr_val)
    cv2.waitKey(0)

    # Save output
    if not args.empty and args.save:
        if not os.path.isdir(args.savepath):
            os.makedirs(args.savepath)
        save_name = args.save + '_{:.4f}.{}'.format(psnr_val, args.save_format)
        cv2.imwrite(os.path.join(args.savepath, save_name), canvas)

def linear_wave(canvas, color=None, contrast=1, gap=5, skew=0, thick=2, rowwise=True):
    mask_shape = list(canvas.shape)
    if rowwise:
        mask_shape[0] += 2 * np.abs(skew)
    else:
        mask_shape[1] += 2 * np.abs(skew)
    mask = np.zeros(mask_shape)
    H, W, _ = mask_shape
    if color is None:
        color = (contrast,) * 3
    if rowwise:
        for row in range(0,H,gap):
            if skew <= 0:
                cv2.line(mask,(0,row),(W,row+skew),color,thickness=thick)
            else:
                cv2.line(mask,(0,row-skew),(W,row),color,thickness=thick)
    else:
        for col in range(0,W,gap):
            if skew <= 0:
                cv2.line(mask,(col,0),(col+skew,H),color,thickness=thick)
            else:
                cv2.line(mask,(col-skew,0),(col,H),color,thickness=thick)
    if rowwise:
        mask = mask[np.abs(skew):-np.abs(skew)]
    else:
        mask = mask[:,np.abs(skew):-np.abs(skew)]

    out = cv2.normalize(canvas+mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return np.uint8(out * 255)

def circles(canvas, centers, max_rad=256, gap=5, thick=2, color=(0,0,0)):
    for center in centers:
        for rad in range(1,max_rad,gap):
            cv2.circle(canvas,center,rad,color,thick)

def lineNSkew(canvas, gap=5, skew=1, color=(0,0,0), thick=1):
    H, W, _ = canvas.shape
    rows = [int(round(val)) for val in np.linspace(0,H,H//gap)]
    for row in rows:
        cv2.line(canvas, (0,row), (W,row), color, thickness=thick)
        cv2.line(canvas, (0,row), (W,row+skew), color, thickness=thick)

def dither(canvas, gap=5, skew=0):
    H, W, _ = canvas.shape
    rows = np.linspace(0,H,H//gap)
    for top_col in range(0,W,gap):
        if not skew:
            cols = [top_col,] * rows.shape[0]
        else:
            cols = np.linspace(top_col,top_col+skew,rows.shape[0])
        for center in zip(cols,rows):
            col,row = tuple([int(round(val)) for val in center])
            cv2.circle(canvas,(col,row),0,(255,0,0))
            cv2.circle(canvas,(col+1,row),0,(0,255,0),-1)
            cv2.circle(canvas,(col,row+1),0,(0,0,255),-1)

def radialShape(canvas, center, radius, count, color=(0,0,0), thick=1):
    gap_theta = np.linspace(0, np.pi * 2, count)
    offsets = [polar2cart(radius, theta) for theta in gap_theta]
    outmosts = [(center[0]+int(off_x), center[1]+int(off_y)) for (off_x, off_y) in offsets]
    for out in outmosts:
        cv2.line(canvas, center, out, color, thickness=thick)

if __name__ == "__main__":
    main()
