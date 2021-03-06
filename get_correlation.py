import argparse
import os
import pickle

import cv2
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", "-i", required=True, help="path to a file containing image file names")
parser.add_argument("--model_heatmaps_dir", "-m", required=True, help="a directory of heatmaps")
parser.add_argument("--fixation_heatmaps_dir", "-x", required=True, action="append", help="a directory of heatmaps (can use multiple times)")
parser.add_argument("--size", "-s", type=int, default=32, help="scale heatmaps to this size")
parser.add_argument("--fig_dir", "-o", help="directory to output comparison figures to")
parser.add_argument("--agg_fig_dir", "-a", type=str, help="output aggregated figures to this path")
parser.add_argument("--img_dir", "-d", help="image directory (only used if --fig_dir passed)")
parser.add_argument("--baseline_weight", "-b", type=float, default=0, help="amount of weight (0 to 1) given to baseline (average) fixation heatmaps")
parser.add_argument("--get_std_err", "-e", action="store_true", help="get correlation standard error by bootstrapping")
parser.add_argument("--get_rand_distr", "-g", action="store_true", help="get distribution (mean and standard deviation) of random correlations by bootstrapping")
opt = parser.parse_args()


def main():
    N = 500  # sample size for bootstrapping
    dim = (opt.size, opt.size)
    with open(opt.input_file) as infile:
        filenames = [line.split()[0] for line in infile.readlines()]

    if opt.baseline_weight > 0:
        print("Computing baseline (average) fixation heatmaps...")
        baseline_heatmaps = [np.zeros(dim) for _ in range(len(opt.fixation_heatmaps_dir))]
        for fn in filenames:
            im_name = os.path.splitext(fn)[0]
            fn = f"{im_name}.npy"
            for i, d in enumerate(opt.fixation_heatmaps_dir):
                path = os.path.join(d, fn)
                hm = np.array(Image.fromarray(np.load(path)).resize(dim))
                hm /= hm.sum()
                baseline_heatmaps[i] += hm
        baseline_heatmaps = [hm / hm.sum() for hm in baseline_heatmaps]

    #plt.matshow(baseline_heatmaps[0])
    #plt.show()

    print("Computing correlations and generating figures...")
    xi = [[] for _ in range(1 + len(opt.fixation_heatmaps_dir))]
    if opt.agg_fig_dir:
        all_figs = []
        all_corrs = []
    for fn in filenames:
        im_name = os.path.splitext(fn)[0]
        fn = f"{im_name}.npy"

        heatmaps = []
        heatmaps_flattened = []

        path = os.path.join(opt.model_heatmaps_dir, fn)
        hm = np.array(Image.fromarray(np.load(path)).resize(dim))
        hm /= hm.sum()
        heatmaps_flattened.append(hm.flatten())
        xi[0].extend(heatmaps_flattened[-1])
        hm -= hm.min()
        hm /= hm.max()
        heatmaps.append(hm)
        for i, d in enumerate(opt.fixation_heatmaps_dir, start=1):
            path = os.path.join(d, fn)
            hm = np.array(Image.fromarray(np.load(path)).resize(dim))
            hm /= hm.sum()
            if opt.baseline_weight > 0:
                gain_weight = 1.0 - opt.baseline_weight
                hm = 1.0/gain_weight * (hm - opt.baseline_weight * baseline_heatmaps[i-1])
                hm /= hm.sum()
            heatmaps_flattened.append(hm.flatten())
            xi[i].extend(heatmaps_flattened[-1])
            hm -= hm.min()
            hm /= hm.max()
            heatmaps.append(hm)

        im_path = os.path.join(opt.img_dir, f"{im_name}.jpg")
        im = imread(im_path)
        im_resized = np.array(Image.fromarray(im).resize(dim)) / 255.0

        fig, axs = plt.subplots(1, len(heatmaps)+1, figsize=(3*(len(heatmaps)+1), 4))
        axs[0].imshow(im / 255.0)

        hm_ref = cv2.cvtColor(cv2.applyColorMap((heatmaps[0] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
        axs[1].imshow(im_resized*0.6 + hm_ref*0.4)
        dir_name = os.path.basename(opt.model_heatmaps_dir)
        axs[1].set_xlabel(dir_name[:dir_name.rfind("_")])

        corrs = []
        for i in range(1, len(heatmaps)):
            hm = cv2.cvtColor(cv2.applyColorMap((heatmaps[i] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
            axs[i+1].imshow(im_resized*0.6 + hm*0.4)
            corr = np.corrcoef(heatmaps_flattened[0], heatmaps_flattened[i])[0, 1]
            corrs.append(corr)
            axs[i+1].set_title(f"Corr: {corr:.3f}")
            dir_name = os.path.basename(opt.fixation_heatmaps_dir[i-1])
            axs[i+1].set_xlabel(dir_name[:dir_name.rfind("_")])

        for ax in axs:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

        if opt.fig_dir or opt.agg_fig_dir:
            fig.suptitle(f"{im_name}")
            plt.tight_layout()

            if opt.fig_dir:
                outpath = os.path.join(opt.fig_dir, f"{im_name}.png")
                plt.savefig(outpath, dpi=100, bbox_inches="tight")

            if opt.agg_fig_dir:
                fig.canvas.draw()
                fig_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                width, height = fig.get_size_inches() * fig.get_dpi()
                width, height = int(width), int(height)
                fig_rgb = fig_rgb.reshape(height, width, 3)
                new_width = int(150 * (2 + len(opt.fixation_heatmaps_dir)))
                new_height = int(new_width / width * height)
                fig_rgb = np.array(Image.fromarray(fig_rgb).resize((new_width, new_height)))
                all_figs.append(fig_rgb)
                all_corrs.append(corrs)

        plt.close()

    if opt.agg_fig_dir:
        # Output aggregated figures sorted by each correlation column
        print("Outputting sorted aggregated figures...")
        for i, d in enumerate(opt.fixation_heatmaps_dir):
            sorted_figs = sorted(enumerate(all_figs), key=lambda tup: all_corrs[tup[0]][i])
            sorted_figs = [tup[1] for tup in sorted_figs]
            im = Image.fromarray(np.vstack(sorted_figs))
            dir_name = os.path.basename(opt.fixation_heatmaps_dir[i])
            map_name = dir_name[:dir_name.rfind("_")]
            outpath = os.path.join(opt.agg_fig_dir, f"{map_name}.png")
            im.save(outpath)
        print("Done.")
        corrs_path = os.path.join(opt.agg_fig_dir, "correlations.pkl")
        img_to_corrs = dict(zip(filenames, all_corrs))
        with open(corrs_path, "wb") as outfile:
            pickle.dump(img_to_corrs, outfile)

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
