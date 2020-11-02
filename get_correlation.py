import os
import argparse

import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", "-i", required=True, help="path to a file containing image file names")
parser.add_argument("--dir1", "-a", required=True, help="first directory of heatmaps")
parser.add_argument("--dir2", "-b", required=True, help="second directory of heatmaps")
opt = parser.parse_args()


def main():
    with open(opt.input_file) as infile:
        filenames = [line.split()[0] for line in infile.readlines()]
    x1 = []
    x2 = []
    for fn in filenames:
        fn = os.path.splitext(fn)[0] + ".npy"
        path1 = os.path.join(opt.dir1, fn)
        path2 = os.path.join(opt.dir2, fn)
        im1 = np.load(path1)
        im2 = np.load(path2)

        if im1.shape[0] > im2.shape[0]:
            im1 = np.array(Image.fromarray(im1).resize(im2.shape))
        elif im2.shape[0] > im1.shape[0]:
            im2 = np.array(Image.fromarray(im2).resize(im1.shape))
        x1.extend(im1.ravel())
        x2.extend(im2.ravel())

    print(np.corrcoef(x1, x2)[0, 1])


if __name__ == "__main__":
    main()
