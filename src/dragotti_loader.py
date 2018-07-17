import os, argparse, re
from itertools import product

from image_tools import psnr, dctII, idctII

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, default='../data/icl_dragotti')
    parser.add_argument('-v', "--verbose", action='store_true',
                        help="Print progress log for initialization of 'DragottiDataset' object")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    dataset = DragottiDataset(args.root_path, args.verbose)

    for key_pair in dataset.recaptured_pairs.keys():
        for idx in dataset.recaptured_pairs[key_pair]:
            filepath = dataset.construct(key_pair, idx)
            #print(filepath, os.path.exists(filepath))

class DragottiDataset(object):
    def __init__(self, root_path, verbose=False):
        super(DragottiDataset, self).__init__()     # in case of potential inheritance

        self.SINGLE = 0
        self.RECAPTURE = 1
        self.MONITOR = 'EA232WMI'       # Invariant
        self._patterns = [r'DS-05-(\d+)-S%(\w+).JPG',
                          r'DS-05-R%(\w+)%(\w+)%(\w+)-(\d+).png']
        self._formats = ['DS-05-{}-S%{}.JPG', 'DS-05-R%{}%{}%{}-{}.png']

        self._dirs = [os.path.join(root_path, dir_name) for dir_name in ['SingleCaptureImages','RecapturedImages']]
        self._cameras = [[item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory,item))] \
                         for directory in self._dirs]
        camera_paths = [[os.path.join(directory,camera) for camera in cameras] \
                                for directory, cameras in zip(self._dirs,self._cameras)]

        categories = ['single','recaptured']
        self._pairs = dict(zip(categories, [dict(),dict()]))

        # Create empty pair map
        for category in categories:
            if category == 'single':
                for camera in self._cameras[0]:
                    self._pairs[category][camera] = list()
            else:
                for camera_pair in product(*self._cameras):    # In (single, recapturd) order
                    for idx in range(len(camera_pair)):
                        self._pairs[category][camera_pair] = list()

        # Fill in the pair map
        for category, paths, pattern in zip(categories, camera_paths, self._patterns):
            for path in paths:
                if verbose:
                    print('-----------------------({}) {}------------------------'.format(category, path))
                for filename in os.listdir(path):
                    match_result = re.findall(pattern,filename)
                    if verbose:
                        print("{}\t\t-> {}".format(filename,match_result))
                    if len(match_result):
                        match_result = match_result[0]
                        if category == 'single':
                            camera = match_result[1].upper()
                            self._pairs[category][camera].append(match_result[0])
                        else:
                            r_camera = match_result[0].upper()
                            s_camera = match_result[2].upper()
                            self._pairs[category][(s_camera,r_camera)].append(match_result[3])

        # Print count
        if verbose:
            print("------------------------------ BEFORE -----------------------------")
            for camera in self._pairs['single'].keys():
                candidates = self._pairs['single'][camera]
                print(camera, len(candidates))
            for pair in self._pairs['recaptured'].keys():
                candidates = self._pairs['recaptured'][pair]
                print(pair, len(candidates))

        # Remove non-overlapping items
        for pair in self._pairs['recaptured'].keys():
            candidates = self._pairs['recaptured'][pair]
            s_camera = pair[self.SINGLE]
            for idx, candidate in enumerate(candidates):
                target = [int(item) for item in self._pairs['single'][s_camera]]
                if int(candidate) not in target:
                    if verbose:
                        print("s_camera: {}, #: {}".format(s_camera, candidate))
                    candidates.pop(idx)

        # Print count
        if verbose:
            print("------------------------------ AFTER -----------------------------")
            for pair in self._pairs['recaptured'].keys():
                candidates = self._pairs['recaptured'][pair]
                print(pair, len(candidates))

    def construct(self, key_pair, idx):
        s_camera, r_camera = key_pair
        filename = self._formats[self.RECAPTURE].format(r_camera, self.MONITOR, s_camera, idx)
        return os.path.join(self._dirs[self.RECAPTURE], r_camera, filename)

    @property
    def recaptured_pairs(self):
        return self._pairs['recaptured']

if __name__ == "__main__":
    main()
