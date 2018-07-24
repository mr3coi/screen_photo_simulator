import numpy as np
import cv2
from module import RecaptureModule
import argparse, os
from image_tools import psnr

margin_help_text = \
'''Width values are in terms of ratios (w.r.t. height or width of the image).
        - 1 value  : width of all margins starting from respective edges. 
        - 2 values : width of {top,bottom} margins and {left,right} margins starting from respective edges
        - 4 values : separate width for each of {top, bottom, left, right} starting from respective edges
                        (Note the order of the edges)
        - 8 values : separate (start, end) pair for each of top, bottom, left, right
                        (the margins begin at 'offset's instead of respective edges)
                     The 'start', 'end' values are ratios w.r.t. height or width
                        starting from the top or the left edges,
                        i.e. a 0.1(10-percent) margin on the right edge would be termed as (0.9, 1),
                        because it "starts" at 0.9(90-percent) location
                        and "ends" at 1.0(100-percent) location w.r.t. the left edge.
'''

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # File I/O
    parser.add_argument("--datapath", default='../../nsml_datasets/w_tower_of_god/train/train/0',
                        help="Path to the directory containing the source image.")
    parser.add_argument("--savepath", default='../data/output/recapture',
                        help="Path to the output storage directory (automatically generated if not yet there).")
    parser.add_argument("--save", type=str, default=None,
                        help="Name of the output file storing the results (not saved if not provided).")
    parser.add_argument("--save-format", type=str, default='jpg',
                        help="File format of the output file (default: JPEG).")

    # Generation parameters
    parser.add_argument("--num-moire", type=int, nargs='+', default=None,
                        help="Number of moires to apply horizontally and vertically (single value is broadcasted). By default, a random integer between 0-2 is chosen for each of horizontal, vertical case.")
    parser.add_argument('-m', '--margins', nargs='+', type=float, default=None,
                        help=margin_help_text)
    parser.add_argument('-g', "--gamma", type=float, default=None,
                        help="Do gamma correction on the given input (default: 1 => no correction)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed value for 'np.random.seed'.")

    # Others
    parser.add_argument("--psnr", action='store_true',
                        help="Compute the PSNR value of the output image.")
    parser.add_argument('-rv', "--recapture-verbose", action='store_true',
                        help="Print the log of progress produced as RecaptureModule transforms the input image.")
    parser.add_argument('-s', "--show", action='store_true',
                        help="Show generated outputs directly.")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Parse number of moire patterns to insert (horizontally & vertically)
    if args.num_moire == None:
        v_moire = np.random.randint(3)
        h_moire = np.random.randint(3)
    elif type(args.num_moire) is list:
        if len(args.num_moire) == 1:
            h_moire = v_moire = args.num_moire[0]
        elif len(args.num_moire) == 2:
            h_moire, v_moire = args.num_moire
        else:
            raise ValueError("Please provide only 1 or 2 'int' values for '--num-moire'.")
    else:
        raise ValueError("Please pass only 1 or 2 'int' values to '--num-moire' parameter.")

    type_dict = dict(); type_dict[0]='f'; type_dict[1]='g'; type_dict[2]='s'

    if h_moire > 0:
        h_type = ''.join([type_dict[np.random.randint(3)] for _ in range(h_moire)])
        h_skew = [np.random.randint(100) for _ in range(h_moire)]
        h_cont = [np.random.randint(20) for _ in range(h_moire)]
        h_dev = [np.random.randint(5) for _ in range(h_moire)]
    else:
        h_type = None
        h_skew = None
        h_cont = None
        h_dev = None
    if v_moire > 0:
        v_type = ''.join([type_dict[np.random.randint(3)] for _ in range(v_moire)])
        v_skew = [np.random.randint(100) for _ in range(v_moire)]
        v_cont = [np.random.randint(20) for _ in range(v_moire)]
        v_dev = [np.random.randint(5) for _ in range(v_moire)]
    else:
        v_type = None
        v_skew = None
        v_cont = None
        v_dev = None

    # Parse margins for warping
    if args.margins is None:
        margins = None
    elif len(args.margins) == 1:
        margins = np.zeros((4,2), dtype="float32")
        margins[0,1] = args.margins[0]
        margins[1,0] = 1-args.margins[0]
        margins[1,1] = 1
        margins[2,1] = args.margins[0]
        margins[3,0] = 1-args.margins[0]
        margins[3,1] = 1
    elif len(args.margins) == 2:
        margins = np.zeros((4,2), dtype="float32")
        margins[0,1] = args.margins[0]
        margins[1,0] = 1-args.margins[0]
        margins[1,1] = 1
        margins[2,1] = args.margins[1]
        margins[3,0] = 1-args.margins[1]
        margins[3,1] = 1
    elif len(args.margins) == 4:
        margins = np.zeros((4,2), dtype="float32")
        margins[0,1] = args.margins[0]
        margins[1,0] = 1-args.margins[1]
        margins[1,1] = 1
        margins[2,1] = args.margins[2]
        margins[3,0] = 1-args.margins[3]
        margins[3,1] = 1
    elif len(args.margins) == 8:
        margins = np.array(args.margins, dtype="float32").reshape((4,2))

    # Parse gamma value
    if args.gamma is not None:
        gamma = args.gamma
    else:
        gamma = np.random.uniform(1,2.5)

    # Generate module as specified
    recap_module = RecaptureModule(v_moire=v_moire, v_type=v_type, v_skew=v_skew, v_cont=v_cont, v_dev=v_dev,
                                   h_moire=h_moire, h_type=h_type, h_skew=h_skew, h_cont=h_cont, h_dev=h_dev,
                                   gamma=gamma,
                                   margins=margins,
                                   seed=args.seed)

    # Load image and apply effects
    for idx, image_name in enumerate(os.listdir(args.datapath)):
        image = cv2.imread(os.path.join(args.datapath, image_name), cv2.IMREAD_COLOR)
        output = recap_module(image, verbose=args.recapture_verbose)
        if args.show:
            cv2.namedWindow('input_{}'.format(idx), cv2.WINDOW_NORMAL)
            cv2.imshow('input_{}'.format(idx), image)
            cv2.namedWindow('output_{}'.format(idx), cv2.WINDOW_NORMAL)
            cv2.imshow('output_{}'.format(idx), output)

        # Save results
        if args.save:
            pass

        if idx == 3:
            break

    if args.show:
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
