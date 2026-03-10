import torch
import torch.nn as nn


class DICELoss(nn.Module):
    def __init__(self,ignore_index):
        super(DICELoss,self).__init__()