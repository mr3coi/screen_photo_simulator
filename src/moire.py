import cv2
import numpy as np

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
                pattern='fixed', contrast=1, color=None, dev=1, seed=0):
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

    out = cv2.normalize(canvas+mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return np.uint8(out * 255)
