import os
import argparse

import cv2
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", "-i", required=True, help="path to a file containing image file names")
parser.add_argument("--dir1", "-a", required=True, help="first directory of heatmaps")
parser.add_argument("--dir2", "-b", required=True, help="second directory of heatmaps")
parser.add_argument("--fig_dir", "-o", help="directory to output comparison figures to")
parser.add_argument("--img_dir", "-m", help="image directory (only used if --fig_dir passed)")
opt = parser.parse_args()


def main():
    with open(opt.input_file) as infile:
        filenames = [line.split()[0] for line in infile.readlines()]
    x1 = []
    x2 = []
    for fn in filenames:
        im_name = os.path.splitext(fn)[0]
        fn = f"{im_name}.npy"
        path1 = os.path.join(opt.dir1, fn)
        path2 = os.path.join(opt.dir2, fn)
        im1 = np.load(path1)
        im2 = np.load(path2)

        if im1.shape[0] > im2.shape[0]:
            im1 = np.array(Image.fromarray(im1).resize(im2.shape))
        elif im2.shape[0] > im1.shape[0]:
            im2 = np.array(Image.fromarray(im2).resize(im1.shape))
        im1 -= im1.min()
        im1 /= im1.sum()
        im2 -= im2.min()
        im2 /= im2.sum()

        im1_raveled = im1.ravel()
        im2_raveled = im2.ravel()
        x1.extend(im1_raveled)
        x2.extend(im2_raveled)

        if opt.fig_dir is not None:
            outpath = os.path.join(opt.fig_dir, f"{im_name}.png")
            corr = np.corrcoef(im1_raveled, im2_raveled)[0, 1]
            im_path = os.path.join(opt.img_dir, f"{im_name}.jpg")
            im = imread(im_path)
            im_resized = np.array(Image.fromarray(im).resize(im1.shape)) / 255.0
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(im / 255.0)
            blk = np.zeros(im1.shape)
            hm1 = cv2.applyColorMap((im1 / im1.max() * 255).astype(np.uint8), cv2.COLORMAP_JET) / 255.0
            hm2 = cv2.applyColorMap((im2 / im2.max() * 255).astype(np.uint8), cv2.COLORMAP_JET) / 255.0
            axs[1].imshow(im_resized*0.6 + hm1*0.4)
            axs[2].imshow(im_resized*0.6 + hm2*0.4)
            fig.suptitle(f"{im_name}\nCorrelation: {corr}")
            plt.tight_layout()
            plt.savefig(outpath, dpi=200, bbox_inches="tight")

    print(np.corrcoef(x1, x2)[0, 1])


if __name__ == "__main__":
    main()
