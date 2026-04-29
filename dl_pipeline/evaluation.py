import os
import glob
import argparse


from yaml import safe_load

import torch
import torch.nn.functional as F

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# from tqdm.notebook import tqdm

import rasterio as rio

import skimage as ski

from model import UNet

FM40_LABELS = [
        91, 92, 93, 98, 99,
        101, 102, 103, 104, 105, 106, 107, 108, 109,
        121, 122, 123, 124,
        141, 142, 143, 144, 145, 146, 147, 148, 149,
        161, 162, 163, 164, 165,
        181, 182, 183, 184, 185, 186, 187, 188, 189,
        201, 202, 203, 204,
    ]

def check_and_make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def load_scene(aef_file, label_file,valid_labels,label_encoder):
    aef_src = rio.open(aef_file)
    label_src = rio.open(label_file)

    aef_arr = aef_src.read()
    label_arr = label_src.read()

    label_arr[~np.isin(label_arr,np.array(valid_labels))] = -1
    label_arr = label_encoder(label_arr)

    aef_tensor = torch.from_numpy(aef_arr.astype(np.float64)).float()
    label_tensor = torch.from_numpy(label_arr.astype(np.int64)).long()

    return aef_tensor, label_tensor

def reassemble_tile(blocks):
    if len(blocks.shape) == 3:
        a = b = int(np.sqrt(blocks.shape[0]))
        h, w = blocks.shape[1], blocks.shape[2]
        tile = np.reshape(np.transpose(np.reshape(blocks,(a,b,h,w)),(0,2,1,3)),(h*a,w*b))

    elif len(blocks.shape) == 4:
        a = b = int(np.sqrt(blocks.shape[0]))
        c = blocks.shape[1]
        h, w = blocks.shape[-2], blocks.shape[-1]
        tile = np.reshape(np.transpose(np.reshape(blocks,(a,b,c,h,w)),(0,3,1,4,2)),(h*a,w*b,c))

    return tile

def compute_metrics(pred_tile,pred_tile_topk,label_tile,classes,label_encoder,label_decoder):
    valid_mask = (label_tile != -1)

    num_classes = len(classes)
    preds, topk, labels = pred_tile[valid_mask], pred_tile_topk[valid_mask],label_tile[valid_mask]

    count = np.bincount(preds*num_classes + labels,minlength=num_classes**2)
    confusion_matrix = np.reshape(count,shape=(num_classes,num_classes))

    acc = accuracy(confusion_matrix)
    topk_acc = topk_accuracy(topk,labels)
    miou, iou, wiou, mwiou, miounz, mwiounz = miou_score(confusion_matrix)

    class_dist = np.sum(confusion_matrix,axis=0) / np.sum(confusion_matrix)

    per_class_ious = {f'iou_{label_decoder(i)}':val for i,val in enumerate(iou)}
    per_class_wious ={f'wiou_{label_decoder(i)}':val for i,val in enumerate(wiou)}
    per_class_dist = {f'cdist_{label_decoder(i)}':val for i,val in enumerate(class_dist)}
    metrics = {
        'acc':acc,
        'topk_acc':topk_acc,
        'miou':miou,
        'mwiou':mwiou,
        'iounz':miounz,
        'mwiounz':mwiounz,
        **per_class_ious,
        **per_class_wious
    }

    return metrics



def accuracy(confusion_matrix):
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

def topk_accuracy(topk,labels):
    in_topk = np.any(labels[:,None] == topk,axis=1)

    topk_acc = np.sum(in_topk) / len(in_topk)

    return topk_acc

def miou_score(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix,axis=1) + np.sum(confusion_matrix,axis=0) - intersection
    iou = ((intersection) / (union + 1e-6))

    class_dist = np.sum(confusion_matrix,axis=0) / np.sum(confusion_matrix)

    mwiou = np.mean(iou*class_dist)
    
    miou = np.mean(iou)

    miounz =np.mean(iou[np.nonzero(union)])
    wiou = iou*class_dist
    mwiounz = np.mean(wiou[np.nonzero(union)])

    return miou, iou, wiou, mwiou, miounz, mwiounz


def main(args):
    exp_name = args.exp_name
    data_root = args.data_root
    ckpt = args.ckpt
    target_pyromes = args.target_pyromes
    results_dir = args.results_dir

    dropped_labels = [91,92,93,98,99]

    valid_labels = [label for label in FM40_LABELS if all([label != drop for drop in dropped_labels])]

    def make_label_encoder(valid_labels):
        label_map = dict(zip(valid_labels,np.arange(len(valid_labels))))
        label_map[-1] = -1
        
        inv_label_map = {v:k for k,v in label_map.items()}

        def encode_fn(x):
            return label_map[x]
        
        def decode_fn(x):
            return inv_label_map[x]
        
        label_encode = np.vectorize(encode_fn)
        label_decode = np.vectorize(decode_fn)
        
        return label_encode, label_decode

    label_encoder, label_decoder = make_label_encoder(valid_labels)

    model = UNet(n_channels=64,n_classes=40)
    model.load_state_dict(torch.load(ckpt,weights_only=True,map_location=torch.device('cpu')))
    model.eval()

    full_results = []

    for pyrome in target_pyromes:
        pyrome_records = []
        pyrome_dir = os.path.join(data_root,f'pyrome_{pyrome}')
        splits = safe_load(open(os.path.join(pyrome_dir,'splits.yml')))

        test_tiles = splits['test']
        print(test_tiles)

        for tile in test_tiles:
            tile_dir = os.path.join(pyrome_dir,tile)

            aef_scene_files = glob.glob(os.path.join(tile_dir,'Scene_*','aef.tif'))
            fbfm40_scene_files = glob.glob(os.path.join(tile_dir,'Scene_*','fbfm40.tif'))

        

            aef_scene_files.sort(key=lambda x: int(x.split('/')[-2].split('_')[-1]))
            fbfm40_scene_files.sort(key=lambda x: int(x.split('/')[-2].split('_')[-1]))

            print(aef_scene_files)

            tile_dset = [load_scene(aef_file,label_file,valid_labels,label_encoder) for aef_file, label_file in zip(aef_scene_files,fbfm40_scene_files)]

            labels = []

            with torch.no_grad():
                input_batch = torch.stack([scene for scene,scene_label in tile_dset],dim=0)

                out = model(input_batch)
                probs = torch.exp(F.log_softmax(out,dim=1))
                model_preds = torch.argmax(probs,dim=1).detach().numpy()
                topk_vals, topk_idxs = torch.topk(probs,k=5,dim=1)

            labels = torch.cat([scene_label for scene,scene_label in tile_dset],dim=0).detach().numpy()
            topk_idxs = topk_idxs.detach().numpy()
            model_pred_tile = reassemble_tile(model_preds)
            label_tile = reassemble_tile(labels)
            topk_tile = reassemble_tile(topk_idxs)

            metrics = compute_metrics(model_pred_tile,topk_tile,label_tile,valid_labels,label_encoder,label_decoder)

            log = {'pyrome':pyrome,'tilenum':tile,**metrics}
            print(log)
            pyrome_records.append(log)

        pyrome_results = pd.DataFrame.from_records(pyrome_records)
        full_results.append(pyrome_results)

    out_dir = os.path.join(results_dir,exp_name)
    check_and_make_dir(out_dir)
    full_results = pd.concat(full_results)
    full_results.to_csv(os.path.join(out_dir,'results.csv'),index=False)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp-name',
        type=str
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./fuels-tiles-10m/'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
    )
    parser.add_argument(
        '--target-pyromes',
        nargs='+'
    )
    parser.add_argument(
        '--results-dir',
        type=str
    )

    args = parser.parse_args()
    main(args)
