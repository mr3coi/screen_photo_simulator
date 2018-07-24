import cv2
from cv2 import getPerspectiveTransform, warpPerspective
import numpy as np
from math import ceil

def dither(canvas, gap=5, skew=0,
           pattern = 'rgb', contrast=255, color=None, rowwise=True):
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
        cols = np.linspace(0,W,W//gap)
        for top_row in range(0,H,gap):
            if not skew:
                rows = [top_row,] * cols.shape[0]
            else:
                rows = np.linspace(top_row,top_row+skew,cols.shape[0])
            for center in zip(rows,cols):
                row,col = tuple([int(round(val)) for val in center])
                if pattern == 'rgb':
                    cv2.circle(mask,(row,col),0,(color[0],0,0),-1)
                    cv2.circle(mask,(row+1,col),0,(0,color[1],0),-1)
                    cv2.circle(mask,(row,col+1),0,(0,0,color[2]),-1)
                elif pattern == 'single':
                    cv2.circle(mask,(row,col),0,color)
                else:
                    raise NotImplementedError()
    else:
        rows = np.linspace(0,H,H//gap)
        for top_col in range(0,W,gap):
            if not skew:
                cols = [top_col,] * rows.shape[0]
            else:
                cols = np.linspace(top_col,top_col+skew,rows.shape[0])
            for center in zip(cols,rows):
                col,row = tuple([int(round(val)) for val in center])
                if pattern == 'rgb':
                    cv2.circle(mask,(col,row),0,(color[0],0,0),-1)
                    cv2.circle(mask,(col+1,row),0,(0,color[1],0),-1)
                    cv2.circle(mask,(col,row+1),0,(0,0,color[2]),-1)
                elif pattern == 'single':
                    cv2.circle(mask,(col,row),0,color)
                else:
                    raise NotImplementedError()

    if rowwise:
        mask = mask[np.abs(skew):-np.abs(skew)]
    else:
        mask = mask[:,np.abs(skew):-np.abs(skew)]

    out = cv2.normalize(canvas+mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return np.uint8(out * 255)

def linear_wave(canvas, gap=5, skew=0, thick=2, rowwise=True,
                pattern='fixed', contrast=1, color=None, dev=1, seed=None):
    '''
    :param pattern: Pattern for the noise to add to each line \
                    ('fixed', 'gaussian', 'sine')
    :type pattern: str
    :param contrast: (mean) value to add to the given image before normalization \
                        (mean for 'gaussian', middle value for 'sine', value for 'fixed'). \
                     The same value is set for all channels; use 'color' to provide different values.
    :type contrast: int
    :param dev: deviation (stddev for 'gaussian', amplitude for 'sine')
    :type dev: float
    '''
    mask_shape = list(canvas.shape)
    if rowwise:
        mask_shape[0] += 2 * np.abs(skew)
    else:
        mask_shape[1] += 2 * np.abs(skew)
    mask = np.zeros(mask_shape)
    H, W, _ = mask_shape
    if color is None:
        color = (contrast,) * 3

    # Generate color map
    num_lines = len(list(range(0,H,gap))) if rowwise else len(list(range(0,W,gap)))
    if pattern=='gaussian':
        if seed:
            np.random.seed(seed)
        color_map = [[int(round(np.clip(np.random.randn() * dev + mean, \
                        max(mean-2*dev,0), min(mean+2*dev,255)))) \
                            for mean in color]
                            for _ in range(num_lines)]
    elif pattern == 'sine':
        color_map = [[int(round(np.clip(np.sin(line_num) * dev + mean, 0, 255))) \
                                for mean in color] \
                            for line_num in range(num_lines)]

    elif pattern == 'fixed':
        color_map = [color,] * num_lines
    else:
        raise NotImplementedError('Please choose a valid pattern type.')

    # Add noise
    if rowwise:
        for row, color in zip(range(0,H,gap), color_map):
            if skew <= 0:
                cv2.line(mask,(0,row),(W,row+skew),color,thickness=thick)
            else:
                cv2.line(mask,(0,row-skew),(W,row),color,thickness=thick)
    else:
        for col, color in zip(range(0,W,gap), color_map):
            if skew <= 0:
                cv2.line(mask,(col,0),(col+skew,H),color,thickness=thick)
            else:
                cv2.line(mask,(col-skew,0),(col,H),color,thickness=thick)
    if rowwise:
        mask = mask[np.abs(skew):-np.abs(skew)]
    else:
        mask = mask[:,np.abs(skew):-np.abs(skew)]

    out = (canvas-mask).clip(0,255)     # Clip instead of normalize
    return np.uint8(out)

    ''' # deprecated (normalize output)
    out = cv2.normalize(canvas+mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return np.uint8(out * 255)
    '''

def nonlinear_wave(canvas, gap=4, skew=0, thick=1, directions='b',
                   pattern='fixed', contrast=7, color=None, dev=3,
                   tb_margins=0, lr_margins=0, seed=None):
    '''
    :param pattern: Pattern for the noise to add to each line \
                    ('fixed', 'gaussian', 'sine')
    :type pattern: str
    :param contrast: (mean) value to add to the given image before normalization \
                        (mean for 'gaussian', middle value for 'sine', value for 'fixed'). \
                     The same value is set for all channels; use 'color' to provide different values.
    :type contrast: int
    :param dev: deviation (stddev for 'gaussian', amplitude for 'sine')
    :type dev: float
    '''
    # Initialize the shape of the mask
    mask_shape = list(canvas.shape)

    # Leave additional space for warping margins
    assert tb_margins >= 0 and tb_margins < 0.5, "Please provide a valid 'tb_margins' value in [0,0.5)."
    assert lr_margins >= 0 and lr_margins < 0.5, "Please provide a valid 'lr_margins' value in [0,0.5)."
    tb_extra = ceil((mask_shape[0] / (1 - 2 * tb_margins) - mask_shape[0]) / 2)
    lr_extra = ceil((mask_shape[1] / (1 - 2 * lr_margins) - mask_shape[1]) / 2)
    mask_shape[0] += 2 * tb_extra
    mask_shape[1] += 2 * lr_extra

    # Check which directions to draw lines in
    rowwise = colwise = False
    if directions == 'b':
        rowwise = True
        colwise = True
    elif directions == 'h':
        rowwise = True
    elif directions == 'v':
        colwise = True
    else:
        raise ValueError("Please provide a valid argument for 'directions' parameter, among {'b','h','v'}.")

    # Leave additional space for full skewing
    if rowwise:
        mask_shape[0] += 2 * np.abs(skew)
    if colwise:
        mask_shape[1] += 2 * np.abs(skew)
    mask = np.zeros(mask_shape)
    H, W, _ = mask_shape

    # Set color
    #contrast = 32
    if color is None:
        color = (contrast,) * 3

    # Draw lines onto mask
    if rowwise:
        ### Generate color map
        num_lines = len(list(range(0,H,gap)))
        if pattern=='gaussian':
            if seed:
                np.random.seed(seed)
            color_map = [[int(round(np.clip(np.random.randn() * dev + mean, \
                            max(mean-2*dev,0), min(mean+2*dev,255)))) \
                                for mean in color]
                                for _ in range(num_lines)]
        elif pattern == 'sine':
            color_map = [[int(round(np.clip(np.sin(line_num) * dev + mean, 0, 255))) \
                                    for mean in color] \
                                for line_num in range(num_lines)]

        elif pattern == 'fixed':
            color_map = [color,] * num_lines
        else:
            raise NotImplementedError('Please choose a valid pattern type.')

        ### Draw lines as specified
        for row, color in zip(range(0,H,gap), color_map):
            if skew <= 0:
                cv2.line(mask,(0,row),(W,row+skew),color,thickness=thick)
            else:
                cv2.line(mask,(0,row-skew),(W,row),color,thickness=thick)
    if colwise:
        ### Generate color map
        num_lines = len(list(range(0,W,gap)))
        if pattern=='gaussian':
            if seed:
                np.random.seed(seed)
            color_map = [[int(round(np.clip(np.random.randn() * dev + mean, \
                            max(mean-2*dev,0), min(mean+2*dev,255)))) \
                                for mean in color]
                                for _ in range(num_lines)]
        elif pattern == 'sine':
            color_map = [[int(round(np.clip(np.sin(line_num) * dev + mean, 0, 255))) \
                                    for mean in color] \
                                for line_num in range(num_lines)]

        elif pattern == 'fixed':
            color_map = [color,] * num_lines
        else:
            raise NotImplementedError('Please choose a valid pattern type.')

        ### Draw lines as specified
        for col, color in zip(range(0,W,gap), color_map):
            if skew <= 0:
                cv2.line(mask,(col,0),(col+skew,H),color,thickness=thick)
            else:
                cv2.line(mask,(col-skew,0),(col,H),color,thickness=thick)

    # Add noise
    if skew:
        if rowwise:
            mask = mask[np.abs(skew):-np.abs(skew)]
        if colwise:
            mask = mask[:,np.abs(skew):-np.abs(skew)]

    # Distort mask
    ### Set the corners of the image as source points
    H, W, _ = mask.shape
    src_points = np.zeros((4,2), dtype="float32")
    src_points[0] = [0,0]    # top-left (w,h)
    src_points[1] = [W-1,0]    # top-right (w,h)
    src_points[2] = [W-1,H-1]    # bottom-right (w,h)
    src_points[3] = [0,H-1]    # bottom-left (w,h)

    ### Randomly generate dest points within the given margins
    t_margin = [0,tb_margins]
    b_margin = [1-tb_margins,1]
    l_margin = [0,lr_margins]
    r_margin = [1-lr_margins,1]
    if seed:
        np.random.seed(seed)
    tl_h, tr_h = np.random.randint(*[H * val for val in t_margin], size=2)
    bl_h, br_h = np.random.randint(*[H * val for val in b_margin], size=2)
    tl_w, bl_w = np.random.randint(*[W * val for val in l_margin], size=2)
    tr_w, br_w = np.random.randint(*[W * val for val in r_margin], size=2)
    dst_points = np.zeros((4,2), dtype="float32")
    dst_points[0] = [tl_w, tl_h]    # top-left
    dst_points[1] = [tr_w, tr_h]   # top-right
    dst_points[2] = [br_w, br_h]   # bottom-right
    dst_points[3] = [bl_w, bl_h]   # bottom-left

    ### Compute warp matrix and warp the mask
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_mask = cv2.warpPerspective(mask, M, (W, H))

    ### Remove (potential) black regions by removing the margins
    warped_mask = warped_mask[tb_extra:-tb_extra, lr_extra:-lr_extra]

    out = (canvas + warped_mask).clip(0,255)
    return np.uint8(out), np.uint8(warped_mask)
