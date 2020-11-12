Deep Learning Attention Mechanisms vs. Human Visual Priors
==========================================================


Show, Attend and Tell
---------------------

PyTorch implementation and pre-trained model taken from [https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

From inside `show-attend-tell/`, run
```
python3 caption.py --img ../doggo.jpg --model pretrained/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map pretrained/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size 5
```


Learn to Pay Attention
----------------------

PyTorch implementation and pre-trained models taken from [https://github.com/SaoYan/LearnToPayAttention](https://github.com/SaoYan/LearnToPayAttention).

From inside `learn-to-pay-attention/`, for paying attention before max-pooling layers, run
```
python3 get_attention_heatmaps.py --img ../doggo.jpg --attn_mode before --model pretrained-before/net.pth --normalize_attn
```
For paying attention after max-pooling layers, run
```
python3 get_attention_heatmaps.py --img ../doggo.jpg --attn_mode after --model pretrained-after/net.pth  --normalize_attn
```


Attention Branch Network
------------------------

PyTorch implementation and pre-trained models taken from [https://github.com/machine-perception-robotics-group/attention_branch_network](https://github.com/machine-perception-robotics-group/attention_branch_network).

From inside `attention-branch-network/`, for the model pre-trained on CIFAR-100, run
```
python3 get_attention_cifar100.py --img ../doggo.jpg --model pretrained-cifar100-resnet110/checkpoint.pth.tar
```
For the model pre-trained on ImageNet, run
```
python3 get_attention_imagenet2012.py --img ../doggo.jpg --model pretrained-imagenet2012-resnet50/checkpoint.pth.tar
```
