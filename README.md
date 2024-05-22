## L-MLP <br> <sub><small>Official PyTorch implementation of [Lateralization MLP: A Simple Brain-inspired Architecture for Diffusion]</small></sub>

The Transformer architecture has dominated machine learning in a wide range of tasks. The specific characteristic of this architecture is an expensive scaled dot-product attention mechanism that models the inter-token interactions, which is known to be the reason behind its success. However, such a mechanism does not have a direct parallel to the human brain which brings the question if the scaled-dot product is necessary for intelligence with strong expressive power. Inspired by the lateralization of the human brain, we propose a new simple but effective architecture called the Lateralization MLP (L-MLP). Stacking L-MLP blocks can generate complex architectures. Each L-MLP block is based on a multi-layer perceptron (MLP) that permutes data dimensions, processes each dimension in parallel, merges them, and finally passes through a joint MLP. We discover that this specific design outperforms other MLP variants and performs comparably to a transformer-based architecture in the challenging diffusion task while being highly efficient. We conduct experiments using text-to-image generation tasks  to demonstrate the effectiveness and efficiency of  L-MLP. Further, we look into the model behavior and discover a connection to the function of the human brain.

--------------------

This codebase implements the L-MLP for diffusion models. Special thanks to [U-ViT](libs/uvit.py) for providing the amazing codebase.


## Dependency

```sh
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116  # install torch-1.13.1
pip install accelerate==0.12.0 absl-py ml_collections einops ftfy==6.1.1 transformers==4.23.1

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. (Perhaps other versions also work, but I haven't tested it.)

## Preparation Before Training and Evaluation

#### Autoencoder
Download `stable-diffusion` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains image autoencoders converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion)). 
Put the downloaded directory as `assets/stable-diffusion` in this codebase.
The autoencoders are used in latent diffusion models.

#### Data
* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Then extract their features according to `scripts/extract_mscoco_feature.py` `scripts/extract_test_prompt_feature.py` `scripts/extract_empty_feature.py`.

#### Reference statistics for FID
Download `fid_stats` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains reference statistics for FID).
Put the downloaded directory as `assets/fid_stats` in this codebase.
In addition to evaluation, these reference statistics are used to monitor FID during the training process.

## Training

We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library to help train with distributed data parallel and mixed precision. The following is the training command:
```sh
# the training setting
num_processes=2  # the number of gpus you have, e.g., 2
train_script=train_t2i_discrete.py  # the train script, one of <train.py|train_ldm.py|train_ldm_discrete.py|train_t2i_discrete.py>
                       # train_t2i_discrete.py: text-to-image training on latent space
config=configs/f2.py  # the training configuration # you can change other hyperparameters by modifying the configuration file

# launch training
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 $train_script --config=$config
```

We provide command to reproduce L-MLP training in the paper:
```sh

# MS-COCO (U-MLP)
accelerate launch --num_processes 1 --mixed_precision fp16 train_t2i_discrete.py --config=configs/lmlp_f2.py

# MS-COCO (U-ViT-S/2) (baseline)
accelerate launch --num_processes 1 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py

```

## Evaluation (Compute FID)

We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library for efficient inference with mixed precision and multiple gpus. The following is the evaluation command:
```sh
# the evaluation setting
num_processes=1  # the number of gpus you have, e.g., 2
eval_script=eval_t2i_discrete.py  
            # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
config=configs/lmlp_f2.py  # the training configuration

# launch evaluation
accelerate launch --num_processes $num_processes --mixed_precision fp16 eval_script --config=$config
```

We provide all commands to reproduce FID results in the paper:
```sh
# MS-COCO (U-ViT-S/2)
accelerate launch --num_processes 1 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=mscoco_uvit_small.pth
accelerate launch --num_processes 1 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/lmlp_f2.py --nnet_path=.\workdir\lmlp_f2\default\ckpts\2000000.ckpt\nnet_ema.pth


## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{bao2022all,
  title={All are Worth Words: A ViT Backbone for Diffusion Models},
  author={Bao, Fan and Nie, Shen and Xue, Kaiwen and Cao, Yue and Li, Chongxuan and Su, Hang and Zhu, Jun},
  booktitle = {CVPR},
  year={2023}
}
```

This implementation is based on
* [Extended Analytic-DPM](https://github.com/baofff/Extended-Analytic-DPM) (provide the FID reference statistics on CIFAR10 and CelebA 64x64)
* [guided-diffusion](https://github.com/openai/guided-diffusion) (provide the FID reference statistics on ImageNet)
* [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (provide the official implementation of FID to PyTorch)
* [dpm-solver](https://github.com/LuChengTHU/dpm-solver) (provide the sampler)
