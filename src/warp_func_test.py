import cv2
import numpy as np
from numpy.linalg import inv
from numpy.linalg.linalg import LinAlgError
import pickle

import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument("--datapath", default='../data/output',
                        help="Path to the directory containing the source image.")
    parser.add_argument("--file", default='linear_single_vertical_gaussian_g1.jpg',
                        help="Name of the source image file.")
    parser.add_argument("--savepath", default='../data/output',
                        help="Path to the output storage directory \
                                (automatically generated if not yet there).")
    parser.add_argument("--save", type=str, default=None,
                        help="Name of the output file storing the results \
                                (not saved if not provided).")
    parser.add_argument("--save-format", type=str, default='jpg',
                        help="File format of the output file (default: JPEG).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed value for 'np.random.seed'.")
    parser.add_argument('-m', "--mode", type=int, default=2,
                        help="The type of test to run.")
    return parser

def warp_image(image):
    pass

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Warp the input image
    # Test I: Transform an image so that a distorted item in the image is recovered

    if args.mode == 1:
        # Load image
        original = cv2.imread(os.path.join(args.datapath,args.file), cv2.IMREAD_COLOR)
        H, W, _ = original.shape

        src_points = np.zeros((4,2), dtype="float32")
        src_points[0] = [165,775]    # top-left
        src_points[1] = [1470,750]    # top-right
        src_points[2] = [1690,1680]    # bottom-right
        src_points[3] = [350,1915]    # bottom-left

        dst_H = 600
        dst_W = 800
        dst_points = np.zeros((4,2), dtype="float32")
        dst_points[0] = [dst_W // 4, dst_H // 4]    # top-left
        dst_points[1] = [dst_W // 4 * 3, dst_H // 4]   # top-right
        dst_points[2] = [dst_W // 4 * 3, dst_H // 4 * 3]   # bottom-right
        dst_points[3] = [dst_W // 4, dst_H // 4 * 3]   # bottom-left

    # Test II: Warp an image

    if args.mode == 2:
        H,W = 600, 800

        ### Generate test image
        test_img = np.zeros((H,W,3), dtype=np.uint8)
        for col in range(0,W,4):
            for row in range(H):
                test_img[row,col,:] = 255
        for row in range(0,H,4):
            for col in range(W):
                test_img[row,col,:] = 255

        ### Set the corners of the image as source points
        src_points = np.zeros((4,2), dtype="float32")
        src_points[0] = [0,0]    # top-left (w,h)
        src_points[1] = [W-1,0]    # top-right (w,h)
        src_points[2] = [W-1,H-1]    # bottom-right (w,h)
        src_points[3] = [0,H-1]    # bottom-left (w,h)

        ### Randomly generate dest points within the given margins
        tb_margins = 0.15
        lr_margins = 0.15
        t_margin = [0,tb_margins]
        b_margin = [1-tb_margins,1]
        l_margin = [0,lr_margins]
        r_margin = [1-lr_margins,1]

        if args.seed:
            np.random.seed(args.seed)
        tl_h, tr_h = np.random.randint(*[H * val for val in t_margin], size=2)
        bl_h, br_h = np.random.randint(*[H * val for val in b_margin], size=2)
        tl_w, bl_w = np.random.randint(*[W * val for val in l_margin], size=2)
        tr_w, br_w = np.random.randint(*[W * val for val in r_margin], size=2)

        dst_points = np.zeros((4,2), dtype="float32")
        dst_points[0] = [tl_w, tl_h]    # top-left
        dst_points[1] = [tr_w, tr_h]   # top-right
        dst_points[2] = [br_w, br_h]   # bottom-right
        dst_points[3] = [bl_w, bl_h]   # bottom-left

        ### Compute warp matrices
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(test_img, M, (W, H))

        # Define windows
        '''
        cv2.namedWindow("original")
        cv2.moveWindow("original",40,80)
        '''
        cv2.namedWindow("warped")
        cv2.moveWindow("warped",40,80)

        # Display result
        #cv2.imshow("original", test_img)
        cv2.imshow("warped", warped)
        cv2.waitKey(0)

        # Save output
        if args.save:
            if not os.path.isdir(args.savepath):
                os.makedirs(args.savepath)
            save_name = args.save + '.' + args.save_format
            cv2.imwrite(os.path.join(args.savepath, save_name), warped)

    # Test III: warping pattern analysis

    if args.mode == 3:
        ### Container for results
        NUM_TESTS = 100
        results = np.zeros((NUM_TESTS, 1+8+1))
        H,W = 600, 800

        ### Generate test image
        test_img = np.zeros((H,W,3), dtype=np.uint8)
        for col in range(0,W,4):
            for row in range(H):
                test_img[row,col,:] = 255
        for row in range(0,H,4):
            for col in range(W):
                test_img[row,col,:] = 255

        ### Set the corners of the image as source points
        src_points = np.zeros((4,2), dtype="float32")
        src_points[0] = [0,0]    # top-left (w,h)
        src_points[1] = [W-1,0]    # top-right (w,h)
        src_points[2] = [W-1,H-1]    # bottom-right (w,h)
        src_points[3] = [0,H-1]    # bottom-left (w,h)

        tb_margins = 0.15
        lr_margins = 0.15
        t_margin = [0,tb_margins]
        b_margin = [1-tb_margins,1]
        l_margin = [0,lr_margins]
        r_margin = [1-lr_margins,1]

        ### Warp the test image
        for it in range(NUM_TESTS):
            ### Randomly generate dest points within the given margins
            np.random.seed(it)
            tl_h, tr_h = np.random.randint(*[H * val for val in t_margin], size=2)
            bl_h, br_h = np.random.randint(*[H * val for val in b_margin], size=2)
            tl_w, bl_w = np.random.randint(*[W * val for val in l_margin], size=2)
            tr_w, br_w = np.random.randint(*[W * val for val in r_margin], size=2)

            dst_points = np.zeros((4,2), dtype="float32")
            dst_points[0] = [tl_w, tl_h]    # top-left
            dst_points[1] = [tr_w, tr_h]   # top-right
            dst_points[2] = [br_w, br_h]   # bottom-right
            dst_points[3] = [bl_w, bl_h]   # bottom-left

            ### Compute warp matrices
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(test_img, M, (W, H))

            test_warped = cv2.warpPerspective(test_img, M, test_img.shape[-2::-1])

            ### Visualize results
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.namedWindow("test_warped", cv2.WINDOW_NORMAL)
            cv2.imshow("test", test_img)
            cv2.imshow("test_warped", test_warped)

            cv2.waitKey(1000)
            decision = int(input("Bizarre enough? (Y:2, N:1) : "))
            results[it] = np.concatenate([np.array([it,]), dst_points.reshape(8), np.array([decision-1,])])

        print(results)
        with open(os.path.join(args.savepath, 'analysis.pkl'), 'wb') as f:
            pickle.dump(results, f)

    # Test IV: Step-by-step morphing

    if args.mode == 4:
        ### Container for results
        NUM_TESTS = 10
        results = np.zeros((NUM_TESTS, 1+8+1))
        H,W = 600, 800

        ### Generate test image
        test_img = np.zeros((H,W,3), dtype=np.uint8)
        for col in range(0,W,4):
            for row in range(H):
                test_img[row,col,:] = 255
        for row in range(0,H,4):
            for col in range(W):
                test_img[row,col,:] = 255

        ### Set the corners of the image as source points
        src_points = np.zeros((4,2), dtype="float32")
        src_points[0] = [0,0]    # top-left (w,h)
        src_points[1] = [W-1,0]    # top-right (w,h)
        src_points[2] = [W-1,H-1]    # bottom-right (w,h)
        src_points[3] = [0,H-1]    # bottom-left (w,h)

        tl_h, tr_h = 0, 0
        bl_h, br_h = H-1, H-1
        tl_w, bl_w = 0, 0
        tr_w, br_w = W-1, W-1

        dst_points = np.zeros((4,2), dtype="float32")

        ### Warp the test image
        for it in range(NUM_TESTS):
            ### Randomly generate dest points within the given margins
            bl_w = int(W * it / 1000 * 4)

            dst_points[0] = [tl_w, tl_h]    # top-left
            dst_points[1] = [tr_w, tr_h]   # top-right
            dst_points[2] = [br_w, br_h]   # bottom-right
            dst_points[3] = [bl_w, bl_h]   # bottom-left

            '''
            ### Compute warp matrices
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            test_warped = cv2.warpPerspective(test_img, M, test_img.shape[-2::-1])

            ### Visualize results
            cv2.namedWindow("test_warped_1_{}".format(it), cv2.WINDOW_NORMAL)
            cv2.imshow("test_warped_1_{}".format(it), test_warped)
            cv2.waitKey(0)
            cv2.destroyWindow("test_warped_1_{}".format(it))
            cv2.waitKey(1)
            '''

        print(">>> Second round")

        for it in range(NUM_TESTS):
            br_w = W - 1 - int(W * it / 1000 * 4)

            dst_points[0] = [tl_w, tl_h]    # top-left
            dst_points[1] = [tr_w, tr_h]   # top-right
            dst_points[2] = [br_w, br_h]   # bottom-right
            dst_points[3] = [bl_w, bl_h]   # bottom-left

            ### Compute warp matrices
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            test_warped = cv2.warpPerspective(test_img, M, test_img.shape[-2::-1])

            ### Visualize results
            cv2.namedWindow("test_warped_2_{}".format(it))
            cv2.moveWindow("test_warped_2_{}".format(it), 40, 80)
            cv2.imshow("test_warped_2_{}".format(it), test_warped)
            cv2.waitKey(0)
            cv2.destroyWindow("test_warped_2_{}".format(it))
            cv2.waitKey(1)

        print(">>> Third round")

        for it in range(NUM_TESTS):
            tr_h = int(H * it / 1000 * 4)

            dst_points[0] = [tl_w, tl_h]    # top-left
            dst_points[1] = [tr_w, tr_h]   # top-right
            dst_points[2] = [br_w, br_h]   # bottom-right
            dst_points[3] = [bl_w, bl_h]   # bottom-left

            ### Compute warp matrices
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            test_warped = cv2.warpPerspective(test_img, M, test_img.shape[-2::-1])

            ### Visualize results
            cv2.namedWindow("test_warped_3_{}".format(it))
            cv2.moveWindow("test_warped_3_{}".format(it), 40, 80)
            cv2.imshow("test_warped_3_{}".format(it), test_warped)
            cv2.waitKey(0)
            cv2.destroyWindow("test_warped_3_{}".format(it))
            cv2.waitKey(1)

        print(">>> Fourth round")

        for it in range(NUM_TESTS):
            tl_w = int(W * it / 1000 * 8)

            dst_points[0] = [tl_w, tl_h]    # top-left
            dst_points[1] = [tr_w, tr_h]   # top-right
            dst_points[2] = [br_w, br_h]   # bottom-right
            dst_points[3] = [bl_w, bl_h]   # bottom-left

            ### Compute warp matrices
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            test_warped = cv2.warpPerspective(test_img, M, test_img.shape[-2::-1])

            ### Visualize results
            cv2.namedWindow("test_warped_4_{}".format(it))
            cv2.moveWindow("test_warped_4_{}".format(it), 40, 80)
            cv2.imshow("test_warped_4_{}".format(it), test_warped)
            cv2.waitKey(0)
            cv2.destroyWindow("test_warped_4_{}".format(it))
            cv2.waitKey(1)

    # Test V: Interactive morphing

    if args.mode == 5:
        ### Container for results
        NUM_TESTS = 10
        results = np.zeros((NUM_TESTS, 1+8+1))
        H,W = 600, 800

        ### Generate test image
        test_img = np.zeros((H,W,3), dtype=np.uint8)
        for col in range(0,W,4):
            for row in range(H):
                test_img[row,col,:] = 255
        for row in range(0,H,4):
            for col in range(W):
                test_img[row,col,:] = 255

        ### Set the corners of the image as source points
        src_points = np.zeros((4,2), dtype="float32")
        src_points[0] = [0,0]    # top-left (w,h)
        src_points[1] = [W-1,0]    # top-right (w,h)
        src_points[2] = [W-1,H-1]    # bottom-right (w,h)
        src_points[3] = [0,H-1]    # bottom-left (w,h)

        tl_h, tr_h = 0, 0
        bl_h, br_h = H-1, H-1
        tl_w, bl_w = 0, 0
        tr_w, br_w = W-1, W-1

        dst_points = np.zeros((4,2), dtype="float32")

        while True:
            ### Change coordinates
            target = int(input("Choose which to manipulate : "))
            if target < 0:
                continue
            elif target == 0:
                break
            else:
                value = int(input("Enter how much to manipulate : "))

            if target == 1:
                tl_h += value
            elif target == 2:
                tl_w += value
            elif target == 3:
                tr_h += value
            elif target == 4:
                tr_w += value
            elif target == 5:
                br_h += value
            elif target == 6:
                br_w += value
            elif target == 7:
                bl_h += value
            elif target == 8:
                bl_w += value

            ### Warp the test image
            dst_points[0] = [tl_w, tl_h]    # top-left
            dst_points[1] = [tr_w, tr_h]   # top-right
            dst_points[2] = [br_w, br_h]   # bottom-right
            dst_points[3] = [bl_w, bl_h]   # bottom-left

            ### Compute warp matrices
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            test_warped = cv2.warpPerspective(test_img, M, test_img.shape[-2::-1])

            ### Visualize results
            cv2.namedWindow("test_warped")
            cv2.moveWindow("test_warped", 40, 60)
            cv2.imshow("test_warped", test_warped)
            cv2.waitKey(0)
            cv2.destroyWindow("test_warped")
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
