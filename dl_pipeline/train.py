import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from model import UNet
from dataset import FM40Fuels
from utils import compute_metrics

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
    '--device',
    type=str,
    default='cuda:0'
)

args = parser.parse_args()

exp_name = args.exp_name
data_root = args.data_root
ignore_labels = args.ignore_labels
log_dir = args.log_dir
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
criterion = args.criterion
device = args.device

#Initialize Cuda
if device == 'cuda' and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

#Build Datasets & Loaders
train_dataset = FM40Fuels(
    root_path=data_root,
    split='train',
    ignore_index=ignore_labels
)

test_dataset = FM40Fuels(
    root_path=data_root,
    split = 'test',
    ignore_labels=ignore_labels
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

label_set = train_dataset.labels()
model = UNet(
    enc_chs=(64,128,256,512,1024),
    dec_chs=(1024, 512, 256, 128, 64),
    num_class=len(label_set),
    out_sz=(160,160)
)

optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
criterion.to(device)

model.to(device)


log_dir = os.path.join(log_dir,exp_name)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
print(f'Logging metrics in {log_dir}')
log_df = pd.DataFrame(columns=['epoch','batch','loss','acc','f1'])

model.train()
for epoch in tqdm(range(epochs),unit='epoch',total=epochs):
    for batch_idx,batch in tqdm(enumerate(train_dataloader),unit='batch',total=len(train_dataloader)):
        optimizer.zero_grad()
        
        image = batch['image']['aef'].to(device)
        target = batch['target'].to(device)

        logit = model(image)
        out = F.softmax(logit,dim=1)

        loss = criterion(out,target)

        metrics = compute_metrics(out,target)

        loss.backward()
        optimizer.step()

        log_str = f'Training Metrics - ACC:{metrics['acc']}  | F1-Macro: {metrics['f1macro']} | mIoU:  {metrics['miou']}'
        
        
        log_record = {'batch':batch_idx,'epoch':epoch,'loss':loss.item(),'acc':metrics['acc'],'f1':metrics['f1']}
        log_df = pd.concat([log_df,pd.DataFrame([log_record])],ignore_index=True)

        if batch_idx % 10 == 0:
            log_df.to_csv(os.path.join(log_dir,'log.csv'),mode='w')


        tqdm.set_postfix(log_str)

