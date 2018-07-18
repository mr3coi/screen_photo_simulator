from moire import linear_wave
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
    def __init__(self,
                 v_moire=0, v_type=None, v_skew=None, v_cont=None, v_dev=None,
                 h_moire=0, h_type=None, h_skew=None, h_cont=None, h_dev=None,
                 gamma=1):
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
        assert gamma > 0, "ERROR: gamma value cannot be 0 or sub-zero."
        self._gamma = gamma

    def __call__(self, image, seed=0, verbose=False):
        '''
        :param image: the input image to transform (HWC format)
        :type image: np.array
        :param seed: seed for pseudorandom generation of 'gaussian' moire patterns (o/w doesn't matter).
        :type seed: int
        :param verbose: whether to print out the processing log or not
        :type verbose: bool

        :return: the image transformed as specified (HWC format)
        :rtype: np.array
        '''
        out = image

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
                                  seed=seed)
                if verbose:
                    print('(Linear moire call) type: {}, skew: {}, contrast: {}, dev: {}, row: {}'.format(
                            mtype, skew, ctr, dev, row))

        # Warping # TODO

        return out

    @property
    def gamma(self):
        return self._gamma
