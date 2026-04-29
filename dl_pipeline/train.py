import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

from model import UNet,UNet10m
from losses import DICELoss,ComboLoss
from dataset import FM40Fuels, FM40Fuels30m
from utils import compute_metrics,evaluate

parser = argparse.ArgumentParser()

parser.add_argument(
    '--exp-name',
    type=str,
    default='test'
)
parser.add_argument(
    '--log-dir',
    type=str,
    default='./logs/'
)
parser.add_argument(
    '--checkpoints',
    type=str,
    default='/home/rdemilt/fuels/eo-fuels-extrapolation/dl_pipeline/checkpoints/'
)
parser.add_argument(
    '--data-root',
    type=str,
    default='/home/rdemilt/fuels/eo-fuels-extrapolation/data/fuels-tiles/'
)
parser.add_argument(
    '--ignore-labels',
    nargs='+',
    type=int,
    default= [91,92,93,98,99]
)
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='Number of iterations over training dataset'
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='Number of images in each training batch'
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-2
)
parser.add_argument(
    '--criterion',
    type=str,
    default='CELoss'
)
parser.add_argument(
    '--input-res',
    type=int,
    default=10
)
parser.add_argument(
    '--device',
    type=str,
    default='cuda:0'
)

args = parser.parse_args()

exp_name = args.exp_name
checkpoint_dir = args.checkpoints
data_root = args.data_root
ignore_labels = args.ignore_labels
log_dir = args.log_dir
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
criterion = args.criterion
res = args.input_res
device = args.device

#Initialize Cuda
if 'cuda' in device and torch.cuda.is_available():
    device = device
else:
    device = 'cpu'

print(f'Training on device {device}')

print('=== Training Parameters ===')
print(f'Experiment Name:{exp_name}')
print(f'Epochs {epochs} | Batch Size {batch_size} | LR {lr}')
print(f'Loss Function: {criterion}')
print(f'Ignoring Labels: {ignore_labels}')

#Build Datasets & Loaders
if res == 30:
    train_dataset = FM40Fuels30m(
        root_path=data_root,
        split='train',
        ignore_index=ignore_labels
    )

    test_dataset = FM40Fuels30m(
        root_path=data_root,
        split = 'test',
        ignore_index=ignore_labels
    )
elif res == 10:
    train_dataset = FM40Fuels(
        root_path=data_root,
        split='train',
        ignore_index=ignore_labels
    )

    test_dataset = FM40Fuels(
        root_path=data_root,
        split = 'test',
        ignore_index=ignore_labels
    )

# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(size=160),
#     transforms.GaussianBlur(kernel_size=3)
# ])


train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

label_set = train_dataset.labels()

if res == 30:
    model = UNet(
        n_channels=64,
        n_classes=len(label_set),
    )
elif res == 10:
    model = UNet10m(
        n_channels=64,
        n_classes=len(label_set),
        in_sz=(480,480),
        out_sz=(160,160)
    )

optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=20,gamma=0.5)
if criterion == 'CELoss':
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
elif criterion == 'DICE':
    criterion = DICELoss(ignore_index=-1)
elif criterion == 'Combo':
    criterion = ComboLoss(ignore_index=-1,alpha=0.5)
criterion.to(device)

model.to(device)


log_dir = os.path.join(log_dir,exp_name)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
print(f'Logging metrics in {log_dir}')
log_df = pd.DataFrame(columns=['epoch','batch','loss','acc','f1'])

model.train()
epoch_pbar = tqdm(range(epochs),unit='epoch',total=epochs)
for epoch in epoch_pbar:
    pbar = tqdm(enumerate(train_dataloader),unit='batch',total=len(train_dataloader),miniters=int(len(train_dataloader) // 10))
    for batch_idx,batch in pbar:
        optimizer.zero_grad()
        
        image = batch['image']['aef'].to(device)
        target = batch['target'].to(device)

        logit = model(image)
        loss = criterion(logit,target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():   
            metrics = compute_metrics(logit,target,label_set)
        
        log_record = {'batch':batch_idx,'epoch':epoch,'loss':loss.item(),'acc':metrics['acc'],'f1':metrics['f1macro'],'weightedf1':metrics['f1weighted']}
        log_df = pd.concat([log_df,pd.DataFrame([log_record])],ignore_index=True)

        if ((batch_idx + 1) % 10) == 0:
            log_df.to_csv(os.path.join(log_dir,'log.csv'),mode='w',index=False)
            pbar.set_postfix(loss=loss.item(),train_acc=metrics['acc'],trainf1=metrics['f1macro'],trainf1weighted=metrics['f1weighted'],miou=metrics['miou'],wiou=metrics['wiou'])
    scheduler.step()

    # evaluate test set
    evaluate(test_dataloader,model,criterion,label_set,epoch_pbar,epoch,device)

    if (epoch + 1) % 25 == 0:
        ckpt_dir = os.path.join(checkpoint_dir,exp_name)
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        outfile = os.path.join(ckpt_dir,f'checkpoint_{(epoch+1)}.pth')

        torch.save(model.state_dict(),outfile)


