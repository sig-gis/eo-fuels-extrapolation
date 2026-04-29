python evaluation.py \
    --exp-name test \
    --data-root /mnt/data/rdemilt/fuels-tiles-30m/ \
    --ckpt ~/spark/fuels/checkpoints/pad-unet-p26/checkpoint_100.pth \
    --target-pyromes 4 27 18 26 45 \
    --results-dir /home/rdemilt/fuels/eo-fuels-extrapolation/dl_pipeline/results/
