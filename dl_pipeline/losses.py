import torch
import torch.nn as nn
import torch.nn.functional as F


class DICELoss(nn.Module):
    def __init__(self,ignore_index):
        super(DICELoss,self).__init__()
        self.ignore_index = ignore_index

    def forward(self,logits,target):
        num_classes = logits.shape[1]

        probs = torch.exp(F.log_softmax(logits,dim=1))

        mask = target != self.ignore_index

        probs = probs * mask.unsqueeze(1)
        
        target = F.one_hot((target*mask).to(torch.long),num_classes=num_classes)
        target =  target.permute(0,3,1,2) * mask.unsqueeze(1)

        intersection = torch.sum((probs * target),dim=(2,3))
        union = torch.sum((probs + target),dim=(2,3))

        dice_score = (2.0 * intersection) / (union + 1e-6)
        
        
        dice_loss = (1.0 - dice_score)

        mask = torch.sum(target,dim=(2,3)) > 0
        dice_loss *= mask.to(dice_loss.dtype)

        return dice_loss.mean()
    
class ComboLoss(nn.Module):

    def __init__(self,ignore_index,alpha=0.5):
        super(ComboLoss,self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

        self.ce_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_fn = DICELoss(ignore_index=ignore_index)
    
    def forward(self,logits,target):

        ce = self.ce_fn(logits,target)
        dice = self.dice_fn(logits,target)

        loss = self.alpha * ce + (1-self.alpha) * dice

        return loss

    