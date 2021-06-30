# Joslim: Joint Widths and Weights Optimization for Slimmable Neural Networks [[PDF]](https://arxiv.org/pdf/2007.11752.pdf)

This repository contains the code for our ECML-PKDD'21 paper [Joslim](https://arxiv.org/abs/2007.11752). This paper improves its previous [workshop versions](https://realworldml.github.io/files/cr/31_PareCO-realml-paper.pdf) in terms of presentation, understanding, and empirical results.

Joslim is an approach to jointly optimize both the width configurations and the shared weights to minimize the area under FLOPs-vs.-loss trade-off curve.

## Disclaimer
This repo is refactored to facilitate the adoption of the code. More specifically, we add comments and restructure the slimmable models. Our paper was based on the [implementation](https://github.com/cmu-enyac/PareCO) that depends on filter pruning code ([LeGR](https://github.com/cmu-enyac/legr)). That version might be hard to build upon if you are interested in different models such as RegNet or EfficientNet (the learning curve is steeper without proper documentation).

To enhance extensibility, we improve our implementation for models and provide a walk-through on how to write a new one that can directly use our joslim code. All details for writing a new model is under `model` folder, or [here](https://github.com/cmu-enyac/Joslim/model).

## Why should I use it?
- If you want to obtain a once-for-all model that can be filter-pruned without re-training, this is for you.

- If you want to have a model to be dynamically pruned in runtime without re-training, this is for you.

- If you want a fast exploration for filter pruning to obtain a family of models with different speed and accuracy, this is for you.

- If you want to build on top of our research such as expanding the search space to achieve even better results, this is for you.

The output of the optimization procedure gives you a set of model configurations and their shared weights. All of them form a trade-off curve such as the one below:

## How to use it?

### Training Joslim-* on ImageNet
To achieve fast training on ImageNet, we leverages `lmdb` from [Image2LMDB](https://github.com/Fangyh09/Image2LMDB). Once you've properly configured the dataset, run the following

**MobileNetV2**

    ./distributed_train.sh 8 --name joslim_mbv2 --dataset ImageNet --epochs 250 --warmup 5 --baselr 0.125 --tau 625 --wd 4e-5 --label_smoothing 0.1 --batch_size 1024 --lower_channel 0.42 --num_sampled_arch 2 --baseline -3 --print_freq 100 --prior_points 20 --scheduler linear_decay --slim_dataaug --scale_ratio 0.25 --datapath YOUR_LMDB_IMAGENET_PATH --network slim_mobilenetv2

**MobileNetV3_Large**

    ./distributed_train.sh 8 --name joslim_mbv3 --dataset ImageNet --epochs 250 --warmup 5 --baselr 0.125 --tau 625 --wd 4e-5 --label_smoothing 0.1 --batch_size 1024 --lower_channel 0.42 --num_sampled_arch 2 --baseline -3 --print_freq 100 --prior_points 20 --scheduler linear_decay --slim_dataaug --scale_ratio 0.08 --datapath YOUR_LMDB_IMAGENET_PATH --network slim_mobilenetv3_large

**ResNet18**

    ./distributed_train.sh 8 --name joslim_res18 --dataset ImageNet --epochs 100 --warmup 5 --baselr 0.125 --tau 250 --wd 4e-5 --label_smoothing 0.1 --batch_size 1024 --lower_channel 0.42 --num_sampled_arch 2 --baseline -3 --print_freq 100 --prior_points 20 --scheduler linear_decay --slim_dataaug --scale_ratio 0.25 --datapath YOUR_LMDB_IMAGENET_PATH --network slim_resnet18

Note that `--tau` interacts with `--epochs`, `--batch_size`, and `--num_sampled_arch` to determine the number of total architectures visited. More specifically, <img src="https://render.githubusercontent.com/render/math?math=num\_total\_arch=\frac{\frac{num\_images}{batch\_size}\times epochs}{tau} \times num\_sampled\_arch">. In the paper, we use 1000 for the number of architectures visited. We use the above equation to figure out `--tau` given all others.

See `model/__init__.py` for currently supported models (`--network` above).

### Training Joslim-* on CIFAR-100
For CIFAR-100, we have `ResNet20`, `ResNet32`, `ResNet44`, and `ResNet56`.


    ./distributed_train.sh 1 --name joslim_res56 --dataset CIFAR100 --epochs 300 --warmup 5 --baselr 0.1 --tau 235 --wd 5e-4 --batch_size 128 --lower_channel 0.42 --num_sampled_arch 2 --baseline -3 --print_freq 100 --prior_points 20 --scheduler linear_decay --datapath YOUR_CIFAR100_PATH --network slim_resnet56


Note that the number from the paper uses 200 epochs, but we later find more epochs are still beneficial.

### Evaluating your trained results
Use the same `--name` as the training time configuration for validation! (We store models based on `--name` and we will retrieve the right one accordingly.)

**MobileNetV2 ImageNet**

    python eval_checkpoints.py --name joslim_mbv2 --datapath YOUR_LMDB_IMAGENET_PATH --dataset ImageNet --network slim_mobilenetv2 --batch_size 1024 --lower_channel 0.42 --slim_dataaug --scale_ratio 0.25

After which you will find `results/joslim_mbv2_eval_pareto.txt` containing the FLOPS (relative to the unpruned model), TOP1, and TOP5 accuracy.

**ResNet56 CIFAR100**

    python eval_checkpoints.py --name joslim_res56 --datapath YOUR_CIFAR100_PATH --dataset CIFAR100 --network slim_resnet56 --batch_size 1024 --lower_channel 0.42

After which you will find `results/joslim_res56_eval_pareto.txt` containing the FLOPS (relative to the unpruned model), TOP1, and TOP5 accuracy.


## Citation

Please consider citing our work if you find this repository useful!

    @inproceedings{chin2021joslim,
		title={Joslim: Joint Widths and Weights Optimization for Slimmable Neural Networks},
		author={Chin, Ting-Wu and Morcos, Ari S and Marculescu, Diana},
		booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
		year={2021},
		organization={Springer}
	}
