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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")

parser.add_argument("--image_dir", "-i", required=True, help="path to images")
parser.add_argument("--num_images", "-n", type=int, required=True, help="Get the most confident n images")
parser.add_argument("--model", "-m", required=True, help="path to model")
parser.add_argument("--attn_mode", type=str, default="before", help="insert attention modules before OR after maxpooling layers")

parser.add_argument("--normalize_attn", action="store_true", help="if True, attention map is normalized by softmax; otherwise use sigmoid")

opt = parser.parse_args()


def main():
    im_size = 32
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    transform_test = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    print('done')

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

    results = []
    with torch.no_grad():
        for img_file in os.scandir(opt.image_dir):
            ## load image
            img = imread(img_file.path)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = np.array(Image.fromarray(img).resize((im_size, im_size)))
            orig_img = img.copy()
            img = img.transpose(2, 0, 1)
            img = img / 255.
            img = torch.FloatTensor(img).to(device)
            image = transform_test(img)  # (3, 256, 256)

            batch = image[np.newaxis, :, :, :]
            pred, __, __, __ = model(batch)
            results.append((torch.max(F.softmax(pred, dim=1)).item(), img_file.name))


    sorted_results = sorted(results, reverse=True)
    print("\n".join(f"{result[1]} {result[0]}" for result in sorted_results[:opt.num_images]))


if __name__ == "__main__":
    main()
