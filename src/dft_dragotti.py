import numpy as np
import cv2
import argparse, os

from random import choice

from image_tools import *
from dragotti_loader import DragottiDataset

def get_parser():
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument("--drag-root", type=str, default='../data/icl_dragotti')
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
    parser.add_argument('-v', "--verbose", action='store_true', help="Print detailed progress log")
    parser.add_argument('-dv', "--drag-verbose", action='store_true',
                        help="Print detailed progress log for initialization of DragottiDataset")
    parser.add_argument("--show", action='store_true', help="Show images")
    parser.add_argument('-r', "--random", action='store_true', help="Choose an image randomly from Dragotti dataset")
    parser.add_argument('-g', "--gamma", type=float, default=1,     # 1/2.2
                        help="Do gamma correction on the given input (default: 1 => no correction)")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    drag = DragottiDataset(args.drag_root, args.drag_verbose)

    # Make resizable windows (as original images are too large)
    if args.show:
        cv2.namedWindow("single", cv2.WINDOW_NORMAL)
        cv2.namedWindow("recaptured", cv2.WINDOW_NORMAL)
    cv2.namedWindow("residual_R", cv2.WINDOW_NORMAL)
    cv2.namedWindow("residual_G", cv2.WINDOW_NORMAL)
    cv2.namedWindow("residual_B", cv2.WINDOW_NORMAL)
    cv2.namedWindow("out_img", cv2.WINDOW_NORMAL)

    src_img = cv2.imread(os.path.join(args.datapath, args.file), cv2.IMREAD_COLOR)
    out_img = []

    for key_pair, indices in drag.recaptured_table.items():
        s_camera, r_camera = key_pair   # TODO sample (currently always uses the first pair)
        if args.random:
            sample_idx = choice(indices)    # Sample an image to test
        else:
            sample_idx = indices[0]    # Sample an image to test

        s_path = drag.s_construct_path(s_camera, sample_idx)
        r_path = drag.r_construct_path(key_pair, sample_idx)
        single_img = cv2.imread(s_path, cv2.IMREAD_COLOR)
        recaptured_img = cv2.imread(r_path, cv2.IMREAD_COLOR)
        if args.gamma != 1:
            recaptured_img = gamma_correction(recaptured_img, gamma=args.gamma)
        single_img = cv2.resize(single_img, recaptured_img.shape[-2::-1])

        if args.show:
            cv2.imshow("single", single_img)
            cv2.imshow("recaptured", recaptured_img)

        residuals = []

        for channel, s_layer, r_layer in zip('RGB', single_img.transpose((2,0,1)),
                                             recaptured_img.transpose((2,0,1))):
            single_dct = dctII(s_layer)
            recaptured_dct = dctII(r_layer)
            residual = recaptured_dct - single_dct
            residuals.append(residual)
            #residual_viz = visualize_dct(residual)
            #residual_viz = histogram_equalize(residual)
            #print(residual_viz)
            #cv2.imshow("residual_{}".format(channel), residual_viz)
            cv2.imshow("residual_{}".format(channel), idctII(residual))

        for src_layer, residual_dct_layer in zip(src_img.transpose((2,0,1)), residuals):
            out_img.append(idctII(dctII(src_layer) + cv2.resize(residual_dct_layer,src_layer.shape[::-1])))
        out_img = np.array(out_img).transpose((1,2,0))
        cv2.imshow("out_img", out_img)

        cv2.waitKey(0)
        break

if __name__ == "__main__":
    main()
