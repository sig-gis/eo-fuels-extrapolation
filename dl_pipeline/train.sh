#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --gpus=1

conda activate fuels

python train.py \
    --exp-name test \
    --log-dir ./logs \
    --data-root /home/rdemilt/fuels/eo-fuels-extrapolation/data/fuels-tiles/ \
    --ignore-labels 91 92 93 98 99 \
    --epochs  1 \
    --batch-size 64 \
    --lr 1e-2 \
    --criterion CELoss \
    --device cuda:0 \
    >log.txt


