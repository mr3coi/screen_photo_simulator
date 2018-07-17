import numpy as np
import cv2
import argparse, os

from random import choice

from image_tools import dctII, idctII
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
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    drag = DragottiDataset(args.drag_root, args.drag_verbose)
    for key_pair, indices in drag.recaptured_table.items():
        s_camera, r_camera = key_pair
        sample_idx = choice(indices)
        s_path = drag.s_construct_path(s_camera, sample_idx)
        r_path = drag.r_construct_path(key_pair, sample_idx)
        print(s_path, os.path.exists(s_path))
        print(r_path, os.path.exists(r_path))

        single_img = cv2.imread(s_path, cv2.IMREAD_COLOR)
        recaptured_img = cv2.imread(r_path, cv2.IMREAD_COLOR)
        print(single_img.shape)
        print(recaptured_img.shape)
        cv2.imshow("single", single_img)
        cv2.imshow("recaptured", recaptured_img)
        cv2.waitKey(0)
        break

if __name__ == "__main__":
    main()
