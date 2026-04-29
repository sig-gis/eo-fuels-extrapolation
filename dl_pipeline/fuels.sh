#!/bin/sh
#SBATCH --job-name pad-unet-ce-p26-10m-max-pool
#SBATCH --nodelist=dragon04
#SBATCH --output=/home/rdemilt/fuels/eo-fuels-extrapolation/dl_pipeline/logs/%x/%j.out

uname -m

conda init
conda activate fuels
python train.py \
    --exp-name %x \
    --log-dir ./logs \
    --checkpoints /home/rdemilt/fuels/eo-fuels-extrapolation/dl_pipeline/checkpoints/ \
    --data-root /home/rdemilt/fuels/eo-fuels-extrapolation/data/fuels-tiles/pyrome_26/ \
    --ignore-labels 91 92 93 98 99 \
    --epochs  100 \
    --batch-size 32 \
    --lr 1e-2 \
    --criterion CELoss \
    --device cuda:0 \
    --input-res 10 \
    >log.txt
