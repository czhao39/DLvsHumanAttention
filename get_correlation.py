import os
import argparse

import cv2
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", "-i", required=True, help="path to a file containing image file names")
parser.add_argument("--dir", "-d", required=True, action="append", help="a directory of heatmaps (use mutliple times for comparing heatmaps)")
parser.add_argument("--size", "-s", type=int, default=32, help="scale heatmaps to this size")
parser.add_argument("--fig_dir", "-o", help="directory to output comparison figures to")
parser.add_argument("--img_dir", "-m", help="image directory (only used if --fig_dir passed)")
parser.add_argument("--get_std_err", "-e", action="store_true", help="get correlation standard error by bootstrapping")
parser.add_argument("--get_rand_distr", "-g", action="store_true", help="get distribution (mean and standard deviation) of random correlations by bootstrapping")
opt = parser.parse_args()


def main():
    N = 500  # sample size for bootstrapping
    dim = (opt.size, opt.size)
    with open(opt.input_file) as infile:
        filenames = [line.split()[0] for line in infile.readlines()]
    xi = [[] for _ in range(len(opt.dir))]
    for fn in filenames:
        im_name = os.path.splitext(fn)[0]
        fn = f"{im_name}.npy"

        heatmaps = []
        heatmaps_raveled = []
        for i, d in enumerate(opt.dir):
            path = os.path.join(d, fn)
            hm = np.array(Image.fromarray(np.load(path)).resize(dim))
            hm /= hm.sum()
            heatmaps_raveled.append(hm.ravel())
            xi[i].extend(heatmaps_raveled[-1])
            hm -= hm.min()
            hm /= hm.max()
            heatmaps.append(hm)

        if opt.fig_dir is not None:
            outpath = os.path.join(opt.fig_dir, f"{im_name}.png")
            im_path = os.path.join(opt.img_dir, f"{im_name}.jpg")
            im = imread(im_path)
            im_resized = np.array(Image.fromarray(im).resize(dim)) / 255.0

            fig, axs = plt.subplots(1, len(heatmaps)+1, figsize=(3*(len(heatmaps)+1), 4))
            axs[0].imshow(im / 255.0)

            hm_ref = cv2.cvtColor(cv2.applyColorMap((heatmaps[0] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
            axs[1].imshow(im_resized*0.6 + hm_ref*0.4)
            dir_name = os.path.basename(opt.dir[0])
            axs[1].set_xlabel(dir_name[:dir_name.rfind("_")])

            for i in range(1, len(heatmaps)):
                hm = cv2.cvtColor(cv2.applyColorMap((heatmaps[i] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
                axs[i+1].imshow(im_resized*0.6 + hm*0.4)
                corr = np.corrcoef(heatmaps_raveled[0], heatmaps_raveled[i])[0, 1]
                axs[i+1].set_title(f"Corr: {corr:.3f}")
                dir_name = os.path.basename(opt.dir[i])
                axs[i+1].set_xlabel(dir_name[:dir_name.rfind("_")])

            for ax in axs:
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])

            fig.suptitle(f"{im_name}")
            plt.tight_layout()
            plt.savefig(outpath, dpi=200, bbox_inches="tight")
            plt.close()

    xi = [np.array(xii) for xii in xi]

    im_size = dim[0] * dim[1]
    for i in range(1, len(xi)):
        corr = np.corrcoef(xi[0], xi[i])[0, 1]
        if opt.get_std_err:
            corrs = []
            for _ in range(N):
                inds = np.random.randint(len(xi[0]), size=len(xi[0]))
                corrs.append(np.corrcoef(xi[0][inds], xi[i][inds])[0, 1])
            std = np.std(corrs)
            print(f"Corr: {corr} ({std})")
        else:
            print(f"Corr: {corr}")

        if opt.get_rand_distr:
            corrs = []
            for _ in range(N):
                shuffled = []
                for j in range(0, len(xi[i]), im_size):
                    xii_shuffled = xi[i][j:j+im_size].copy()
                    np.random.shuffle(xii_shuffled)
                    shuffled.extend(xii_shuffled)
                corrs.append(np.corrcoef(xi[0], shuffled)[0, 1])
            mean = np.mean(corrs)
            std = np.std(corrs)
            z = (corr - mean) / std
            print(f"Z: {z}, Mean: {mean}, Std Dev: {std}\n")


if __name__ == "__main__":
    main()
