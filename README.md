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

PyTorch implementation and pre-trained model taken from [https://github.com/SaoYan/LearnToPayAttention](https://github.com/SaoYan/LearnToPayAttention).

From inside `learn-to-pay-attention/`, run
```
python3 show_attention.py --img ../doggo.jpg --model before-pretrained/net.pth  --normalize_attn
```
