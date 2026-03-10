import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score,f1_score

def accuracy(preds,labels,valid_mask):

    preds = torch.flatten(preds)[valid_mask.flatten()]
    labels = torch.flatten(labels)[valid_mask.flatten()]

    return torch.mean(preds==labels)

def weighted_accuracy(preds,labels,class_weights):

    return

def f1score(preds,labels,valid_mask,classes):
    num_classes = len(classes)
    preds = torch.flatten(preds)[valid_mask.flatten()]
    labels = torch.flatten(labels)[valid_mask.flatten()]

    label_vals, label_counts = torch.unique(labels,return_counts=True)
    label_support = label_counts / labels.size()

    preds_one_hot = F.one_hot(preds,num_classes=num_classes)
    labels_one_hot = F.one_hot(labels,num_classes=num_classes)

    #macro f1
    tp = torch.mean(preds_one_hot & labels_one_hot,dim=0)
    fp = torch.mean(preds_one_hot & ~labels_one_hot,dim=0)
    fn = torch.mean(~preds_one_hot & labels_one_hot,dim=0)

    f1_macro = torch.mean((2*tp) / ((2*tp) + fp + fn))

    f1_weighted = torch.mean(((2*tp) / ((2*tp) + fp + fn)) * label_support)

    #micro f1
    tp = torch.mean(preds_one_hot & labels_one_hot)
    fp = torch.mean(preds_one_hot & ~labels_one_hot)
    fn = torch.mean(~preds_one_hot & labels_one_hot)

    f1_micro = (2*tp) / ((2*tp) + fp + fn)
    
    return f1_micro, f1_macro, f1_weighted

def miou_score(confusion_matrix):
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
    iou = (intersection / (union + 1e-6)) * 100

    miou = torch.mean(iou)
    return miou

def dice(pred,label):
    return

def compute_metrics(out,label,classes):
    predictions = torch.argmax(out,dim=1).squeeze(dim=1)

    valid_mask = (label != -1)

    num_classes = len(classes)
    preds,labels = predictions[valid_mask], label[valid_mask]

    count = torch.bincount(preds*num_classes + labels)

    confusion_matrix = count.view(num_classes,num_classes)

    acc = accuracy(predictions,label,valid_mask)
    f1_micro, f1_macro, f1_weighted = f1score(predictions,label,valid_mask)
    miou = miou_score(confusion_matrix)

    metrics = {
        'acc':acc.item(),
        'f1micro':f1_micro.item(),
        'f1macro':f1_macro.item(),
        'f1weighted':f1_weighted.item(),
        'miou':miou.item()
    }

    return metrics