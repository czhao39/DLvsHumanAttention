"""
Takes in eye fixation data in the format used at https://data.mendeley.com/datasets/8rj98pp6km/1, and converts to heatmaps.
"""

import os
import argparse

from imageio import imread
import numpy as np
from sklearn.neighbors import KernelDensity


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", "-i", required=True, help="path to fixation file")
parser.add_argument("--image_dir", "-m", required=True, help="path to input images, just to get image dimensions")
parser.add_argument("--output_dir", "-o", required=True, help="path to output heatmaps")
opt = parser.parse_args()


def main():
    stddev = 20.0

    data = np.loadtxt(opt.input_file, skiprows=1)
    data = data[~np.isnan(data).any(axis=1)].astype(int)

    for img_ind in np.unique(data[:, 3]):
        img_ind = int(img_ind)
        fixations = data[data[:, 3] == img_ind, 0:2]
        kde = KernelDensity(bandwidth=stddev)
        kde.fit(fixations)

        img_path = os.path.join(opt.image_dir, f"image_r_{img_ind}.jpg")
        img = imread(img_path)
        X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        xy = np.vstack([X.ravel(), Y.ravel()]).T

        heatmap = np.exp(kde.score_samples(xy)).reshape(img.shape[0], img.shape[1])
        out_path = os.path.join(opt.output_dir, f"image_r_{img_ind}_fixationheatmap.npy")
        np.save(out_path, heatmap)

if __name__ == "__main__":
    main()
