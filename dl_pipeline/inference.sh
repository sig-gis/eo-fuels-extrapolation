python inference.py \
    --exp-name p45-inference \
    --data-root /mnt/data/rdemilt/fuels-tiles-30m/ \
    --ckpt ~/spark/fuels/checkpoints/pad-unet-ce-p45/checkpoint_100.pth \
    --target-pyromes 4 27 18 26 45 \
    --results-dir /home/rdemilt/fuels/eo-fuels-extrapolation/dl_pipeline/results/ \
    --n-top-k 5