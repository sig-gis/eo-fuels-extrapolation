import torch
import torch.nn.functional as F

import numpy as np 

from tqdm import tqdm

from tabulate import tabulate

def evaluate(test_dataloader,model,criterion,label_set,epoch_bar,epoch,device):
    model.eval()

    acc = 0
    f1 = 0
    miou = 0
    iou = np.zeros(len(label_set))
    eval_loss = 0
    eval_dist = np.zeros(len(label_set))
    per_class_f1 = np.zeros(len(label_set))
    with torch.no_grad():
        for batch in test_dataloader:
            image = batch['image']['aef'].to(device)
            target = batch['target'].to(device)

            logit = model(image)

            loss = criterion(logit,target)

            metrics = compute_metrics(logit,target,label_set)

            acc += metrics['acc']
            f1 += metrics['f1macro']
            miou += metrics['miou']
            iou += metrics['iou']
            per_class_f1 += metrics['per_class_f1s']
            eval_dist += metrics['dist']
            eval_loss += loss
        
        acc /= len(test_dataloader)
        f1 /= len(test_dataloader)
        miou /= len(test_dataloader)
        eval_dist /= len(test_dataloader)
        eval_loss /= len(test_dataloader)
        iou /= len(test_dataloader)
        per_class_f1 /= len(test_dataloader)

        print(f'=== Epoch {epoch} metrics ===')
        print(tabulate(zip(label_set,list(iou),list(per_class_f1),list(eval_dist)),headers=['Class','mIoU','f1','Class Dist'],tablefmt='pipe',floatfmt='0.5f'))
        print('==============================')

        epoch_bar.set_postfix(eval_loss=loss.item(),acc=acc,f1=f1,miou=miou)
    
    model.train()

def accuracy(confusion_matrix):

    return torch.sum(torch.diag(confusion_matrix)) / torch.sum(confusion_matrix)

def weighted_accuracy(preds,labels,class_weights):

    return

def f1score(confusion_matrix):
    
    tp = torch.diag(confusion_matrix)
    
    precision = tp / (torch.sum(confusion_matrix,dim=0) + 1e-6)
    recall = tp / (torch.sum(confusion_matrix,dim=1) + 1e-6)

    f1 = (2 * (precision * recall)) / (precision + recall + 1e-6)

    weighted_f1 = f1 * (torch.sum(confusion_matrix,dim=0) / (torch.sum(confusion_matrix)))
    f1_macro = torch.mean(f1)
    f1_macro_weighted = torch.mean(weighted_f1)

    return f1_macro, f1_macro_weighted, f1

def miou_score(confusion_matrix):
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
    iou = ((intersection) / (union + 1e-6))

    class_dist = torch.sum(confusion_matrix,dim=0) / torch.sum(confusion_matrix)

    wiou = torch.sum(iou*class_dist)
    
    miou = torch.mean(iou)
    return miou, iou, wiou

def compute_metrics(logits,label,classes):
    probs = torch.exp(F.log_softmax(logits,dim=1))
    predictions = torch.argmax(probs,dim=1).squeeze(dim=1)

    valid_mask = (label != -1)

    num_classes = len(classes)
    preds,labels = predictions[valid_mask], label[valid_mask]

    count = torch.bincount(preds*num_classes + labels,minlength=num_classes**2)

    confusion_matrix = count.view(num_classes,num_classes)

    class_dist = torch.sum(confusion_matrix,dim=0) / torch.sum(confusion_matrix)

    acc = accuracy(confusion_matrix)
    f1_macro, f1_macro_weighted, f1 = f1score(confusion_matrix)
    miou, iou, wiou = miou_score(confusion_matrix)

    metrics = {
        'acc':acc.item(),
        'f1macro':f1_macro.item(),
        'f1weighted':f1_macro_weighted.item(),
        'per_class_f1s':f1.cpu().detach().numpy(),
        'miou':miou.item(),
        'wiou':wiou.item(),
        'iou':iou.cpu().detach().numpy(),
        'dist':class_dist.cpu().detach().numpy()
    }

    return metrics