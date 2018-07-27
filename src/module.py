import numpy as np
from cv2 import getPerspectiveTransform, warpPerspective
from moire import linear_wave, nonlinear_wave
from image_tools import gamma_correction
from warp import warp_image

class RecaptureModule(object):
    '''
    Supports a combination of the following attacks:
    - (Linear) Moire patterns (horizontal & vertical, with skews (in pixels))
    - Gamma correction
    - Warping

    Stores related parameters as member variables to allow for re-use of the module.
    '''
    def __init__(self, dst_H=-1, dst_W=-1,
                 v_moire=0, v_type=None, v_skew=None, v_cont=None, v_dev=None,
                 h_moire=0, h_type=None, h_skew=None, h_cont=None, h_dev=None,
                 nl_moire=False, nl_dir=None, nl_type=None, nl_skew=None, nl_cont=None, nl_dev=None,
                 nl_tb=None, nl_lr=None,
                 gamma=1, points=None, margins=None, seed=None):
        '''
        :param v_moire: # of vertical moire patterns to insert.
        :type v_moire: int
        :param v_type: A sequence of characters 'f'(fixed), 'g'(gaussian), and 's'(sine).
                       Its length may be a factor of 'v_moire' value,
                            in which case the sequence is broadcasted.
        :type v_type: str
        :param v_skew: A list of the numbers of pixels to skew for each moire pattern
                            (in this case how much to skew horizontally).
                       Supports broadcasting.
        :type v_skew: list(int)
        :param v_cont: A list of the values of pixels to add into the input image
                            (before normalization).
                       Determines the strength of the inserted moire pattern
                            ('gaussian': mean, 'sine': center value).
                       Supports broadcasting.
        :type v_cont: list(int)
        :param v_dev: A list of the values of pixels that serve as "deviations"
                            ('gaussian': stddev, 'sine': amplitude).
                      Supports broadcasting.
        :type v_dev: list(int)

        :param h_moire: # of horizontal moire patterns to insert.
        :type h_moire: int
        :param h_type: A sequence of characters 'f'(fixed), 'g'(gaussian), and 's'(sine).
                       Its length may be a factor of 'h_moire' value,
                            in which case the sequence is broadcasted.
        :type h_type: str
        :param h_skew: A list of the numbers of pixels to skew for each moire pattern
                            (in this case how much to skew horizontally).
                       Also supports broadcasting.
        :type h_skew: list(int)
        :param h_cont: A list of the values of pixels to add into the input image
                            (before normalization).
                       Determines the strength of the inserted moire pattern
                            ('gaussian': mean, 'sine': center value).
                       Supports broadcasting.
        :type h_cont: list(int)
        :param h_dev: A list of the values of pixels that serve as "deviations"
                            ('gaussian': stddev, 'sine': amplitude).
                      Supports broadcasting.
        :type h_dev: list(int)
        :param gamma: gamma value for gamma correction (no correction when gamma == 1).
        :type gamma: float

        e.g.)
            recap_module = RecaptureModule(v_moire=2, v_type='sg', v_skew=[20, 80], v_cont=10, v_dev=3,
                                           h_moire=2, h_type='f', h_skew=[20, 80], h_cont=10, h_dev=3,
                                           gamma=2.2)
            output = recap_module(image)
        '''
        # Shape of output if provided
        self._out_H = dst_H; self._out_W = dst_W
        self._seed = seed

        # ================================== Linear Moire pattern ========================================
        type_dict = dict(f='fixed',g='gaussian',s='sine')
        self._counts = []
        self._mtypes = []
        self._skews = []
        self._contrasts = []
        self._devs = []

        for count, mtype, skew, ctr, dev in zip([h_moire, v_moire], [h_type, v_type], [h_skew, v_skew],
                                                [h_cont, v_cont], [h_dev, v_dev]):
            if count > 0:
                if mtype is None or (len(mtype) != count and count % len(mtype) != 0):
                    raise ValueError("Please check the number of your moire type parameters.")
                if len(mtype) != count:
                    mtype *= count // len(mtype)

                if skew is None or (type(skew) is list and len(skew) != count and count % len(skew) != 0):
                    raise ValueError("Please check the number of your skew parameters.")
                if type(skew) is int:
                    skew = [skew,]
                elif type(skew) is not list:
                    raise ValueError("Please check that the argument to the 'skew' parameter is \
                                        either an 'int' or a 'list'.")
                if len(skew) != count:
                    skew *= count // len(skew)

                if ctr is None or (type(ctr) is list and len(ctr) != count and count % len(ctr) != 0):
                    raise ValueError("Please check the number of your contrast parameters.")
                if type(ctr) is int:
                    ctr = [ctr,]
                elif type(ctr) is not list:
                    raise ValueError("Please check that the argument to the 'cont' parameter is \
                                        either an 'int' or a 'list'.")
                if len(ctr) != count:
                    ctr *= count // len(ctr)

                if dev is None or (type(dev) is list and len(dev) != count and count % len(dev) != 0):
                    raise ValueError("Please check the number of your contrast parameters.")
                if type(dev) is int:
                    dev = [dev,]
                elif type(dev) is not list:
                    raise ValueError("Please check that the argument to the 'dev' parameter is \
                                        either an 'int' or a 'list'.")
                if len(dev) != count:
                    dev *= count // len(dev)

                self._counts.append(count)
                self._mtypes.append([type_dict[c] for c in mtype])
                self._skews.append(skew)
                self._contrasts.append(ctr)
                self._devs.append(dev)
            elif count == 0:
                self._counts.append(count)
                self._mtypes.append(list())
                self._skews.append(list())
            else:
                raise ValueError("Count of moire patterns cannot be sub-zero; please check your count parameters.")

        # ================================== Non-linear Moire pattern ========================================
        self._nl_moire  = nl_moire
        self._nl_dir    = nl_dir
        self._nl_type   = nl_type
        self._nl_skew	= nl_skew
        self._nl_cont	= nl_cont
        self._nl_dev	= nl_dev
        self._nl_tb	= nl_tb
        self._nl_lr	= nl_lr

        # ================================== Warping ========================================
        if points is not None:
            self._points = points
            self._warp = 1
        elif margins is not None:
            self._margins = margins
            self._warp = 2
        else:
            self._warp = 0

        # ================================== Gamma correction ========================================
        assert gamma > 0, "ERROR: gamma value cannot be 0 or sub-zero."
        self._gamma = gamma

    def __call__(self, image, new_src_pt=None, new_dst_pt=None, verbose=False, show_mask=False):
        '''
        :param image: the input image to transform (HWC format)
        :type image: np.array
        :param verbose: whether to print out the processing log or not
        :type verbose: bool

        :return: the image transformed as specified (HWC format)
        :rtype: np.array
        '''
        out = image
        H, W, _ = image.shape

        # Gamma correction
        if self._gamma != 1:
            out = gamma_correction(out, gamma=self._gamma)
            if verbose:
                print('(Gamma correction call) gamma: {}'.format(self._gamma))

        # Linear moire pattern insertion
        for row, count, mtypes, skews, conts, devs \
                in zip([True,False], self._counts, self._mtypes, self._skews, self._contrasts, self._devs):
            for it, mtype, skew, ctr, dev in zip(range(count), mtypes, skews, conts, devs):
                out = linear_wave(out,
                                  rowwise=row,
                                  skew=skew,
                                  pattern=mtype,
                                  contrast=ctr,
                                  dev=dev,
                                  seed=self._seed)
                if verbose:
                    print('(Linear moire call) type: {}, skew: {}, contrast: {}, dev: {}, row: {}'.format(
                            mtype, skew, ctr, dev, row))

        # Non-linear moire pattern insertion
        if self._nl_moire:
            out, nl_mask = nonlinear_wave(out, directions=self._nl_dir, pattern=self._nl_type,
                                 skew=self._nl_skew, contrast=self._nl_cont, dev=self._nl_dev,
                                 tb_margins=self._nl_tb, lr_margins=self._nl_lr, seed=self._seed)
            if verbose:
                print('(Non-linear moire call) direction: {}, type: {}, skew: {}, contrast: {}, dev: {}, \
                        margins: {}, {}' \
                        .format(self._nl_dir, self._nl_type, self._nl_skew, self._nl_cont, self._nl_dev,
                                self._nl_tb, self._nl_lr))
        else:
            nl_mask = None

        # Warping
        dst_H = self._out_H if self._out_H > 0 else H
        dst_W = self._out_W if self._out_W > 0 else W

        if new_dst_pt is not None:
            dst_points = new_dst_pt
        else:
            if self._warp == 1:             # Map to predetermined coordinates
                dst_points = self._points
            elif self._warp == 2:           # Map to random coordinates within the given margins
                t_margin, b_margin, l_margin, r_margin = self._margins

                np.random.seed(self._seed)
                tl_h, tr_h = np.random.randint(*[dst_H * val for val in t_margin], size=2)
                bl_h, br_h = np.random.randint(*[dst_H * val for val in b_margin], size=2)
                tl_w, bl_w = np.random.randint(*[dst_W * val for val in l_margin], size=2)
                tr_w, br_w = np.random.randint(*[dst_W * val for val in r_margin], size=2)

                dst_points = np.zeros((4,2), dtype="float32")
                dst_points[0] = [tl_w, tl_h]    # top-left
                dst_points[1] = [tr_w, tr_h]   # top-right
                dst_points[2] = [br_w, br_h]   # bottom-right
                dst_points[3] = [bl_w, bl_h]   # bottom-left

        if new_dst_pt is not None or self._warp:
            if new_src_pt is not None:
                src_points = new_src_pt
            else:
                src_points = np.zeros((4,2), dtype="float32")
                src_points[0] = [0,0]    # top-left (w,h)
                src_points[1] = [W-1,0]    # top-right (w,h)
                src_points[2] = [W-1,H-1]    # bottom-right (w,h)
                src_points[3] = [0,H-1]    # bottom-left (w,h)

            M = getPerspectiveTransform(src_points, dst_points)
            out = warpPerspective(out, M, (dst_W, dst_H))

        if self._nl_moire and show_mask:
            return out, nl_mask
        else:
            return out

    @property
    def gamma(self):
        return self._gamma

    @property
    def out_shape(self):
        '''
        Returns the shape of the output in (H,W) order.
        For both height and width, 'None' indicates that the output is \
                of the same height/width as the input.
        '''
        return (self._out_H if self._out_H is not None else None,
                self._out_W if self._out_W is not None else None)
