import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from model1 import AttnVGG_before
from model2 import AttnVGG_after
from utilities import *

from imageio import imread
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")

parser.add_argument('--img', '-i', help='path to image')
parser.add_argument('--model', '-m', help='path to model')
parser.add_argument("--save", action="store_true", help="if True, output attention maps to files")
parser.add_argument("--attn_mode", type=str, default="before", help='insert attention modules before OR after maxpooling layers')

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')

opt = parser.parse_args()


def main():
    im_size = 32
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    transform_test = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    print('done')


    ## load image
    print('\nloading image ...\n')
    img = imread(opt.img)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize((im_size, im_size)))
    orig_img = img.copy()
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    image = transform_test(img)  # (3, 256, 256)


    ## load network
    print('\nloading the network ...\n')
    # (linear attn) insert attention befroe or after maxpooling?
    # (grid attn only supports "before" mode)
    if opt.attn_mode == 'before':
        print('\npay attention before maxpooling layers...\n')
        net = AttnVGG_before(im_size=im_size, num_classes=100,
            attention=True, normalize_attn=opt.normalize_attn, init='xavierUniform')
    elif opt.attn_mode == 'after':
        print('\npay attention after maxpooling layers...\n')
        net = AttnVGG_after(im_size=im_size, num_classes=100,
            attention=True, normalize_attn=opt.normalize_attn, init='xavierUniform')
    else:
        raise NotImplementedError("Invalid attention mode!")
    print('done')


    ## load model
    print('\nloading the model ...\n')
    state_dict = torch.load(opt.model, map_location=str(device))
    # Remove 'module.' prefix
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    print('done')


    model = net


    if opt.save:
        print("\nwill save heatmaps\n")
        file_prefix = os.path.splitext(os.path.basename(opt.img))[0]


    with torch.no_grad():
        print('\nlog attention maps ...\n')
        # base factor
        if opt.attn_mode == 'before':
            min_up_factor = 1
        else:
            min_up_factor = 2
        # sigmoid or softmax
        if opt.normalize_attn:
            vis_fun = visualize_attn_softmax
        else:
            vis_fun = visualize_attn_sigmoid
        batch = image[np.newaxis, :, :, :]
        __, c1, c2, c3 = model(batch)
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(orig_img)
        if c1 is not None:
            attn1 = vis_fun(img, c1, up_factor=min_up_factor, nrow=1, hm_file=None if not opt.save else file_prefix + "_c1.npy")
            axs[1].imshow(attn1.numpy().transpose(1, 2, 0))
        if c2 is not None:
            attn2 = vis_fun(img, c2, up_factor=min_up_factor*2, nrow=1, hm_file=None if not opt.save else file_prefix + "_c2.npy")
            axs[2].imshow(attn2.numpy().transpose(1, 2, 0))
        if c3 is not None:
            attn3 = vis_fun(img, c3, up_factor=min_up_factor*4, nrow=1, hm_file=None if not opt.save else file_prefix + "_c3.npy")
            axs[3].imshow(attn3.numpy().transpose(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    main()

