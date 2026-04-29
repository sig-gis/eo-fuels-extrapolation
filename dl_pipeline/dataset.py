import os
import glob
from yaml import safe_load

import torch
from torch.utils.data import Dataset

import numpy as np
import xarray as xar

class FM40Fuels(Dataset):
    FM40_LABELS = [
        91, 92, 93, 98, 99,
        101, 102, 103, 104, 105, 106, 107, 108, 109,
        121, 122, 123, 124,
        141, 142, 143, 144, 145, 146, 147, 148, 149,
        161, 162, 163, 164, 165,
        181, 182, 183, 184, 185, 186, 187, 188, 189,
        201, 202, 203, 204,
    ]


    def __init__(self,
        root_path='../data/pyrome_26/',
        im_size=160,
        ignore_index=[91,92,93,98,99],
        split='train',
        train_tiles = [
                            '00439', '00433', '00371', '00372', '00370', '00436', '00500', \
                            '00369', '00501', '00434', '00502', '00437', '00367', '00306', \
                            '00435', '00368', '00503', '00366', '00305', '00438', '00432'
                    ],
        transform = None
    ):
        
        self.root_path=root_path
        self.im_size = im_size
        self.ignore_index = ignore_index
        self.split = split

       
        if isinstance(self.train_tiles,list):
            self.train_tiles = train_tiles
        elif isinstance(self.train_tiles,str):
            splits = safe_load(open(train_tiles))
            self.train_tiles = splits['train']
            

        self.transform = transform

        tile_size=1600
        self.num_scenes = (tile_size // self.im_size)**2

        self.tilenums = glob.glob(os.path.join(root_path,'*',''))
        self.valid_labels = [label for label in self.FM40_LABELS if not any(label == ignored for ignored in self.ignore_index)]
        self.label_encode = self.make_label_encoder()


        self.aef_files = sorted(glob.glob(os.path.join(root_path,'*','Scene_*','aef.zarr')))
        self.fbfm40_files = sorted(glob.glob(os.path.join(root_path,'*','Scene_*','fbfm40.zarr')))

        if self.split =='train':
            self.aef_files = [file for file in self.aef_files if any([tilenum in file for tilenum in self.train_tiles])]
            self.fbfm40_files = [file for file in self.fbfm40_files if any([tilenum in file for tilenum in self.train_tiles])]
        elif self.split =='test':
            self.aef_files = [file for file in self.aef_files if not any([tilenum in file for tilenum in self.train_tiles])]
            self.fbfm40_files = [file for file in self.fbfm40_files if not any([tilenum in file for tilenum in self.train_tiles])]

        

    def make_label_encoder(self):
        self.label_map = dict(zip(self.valid_labels,np.arange(len(self.valid_labels))))
        self.label_map[-1] = -1
        

        def encode_fn(x):
            return self.label_map[x]
        
        label_encode = np.vectorize(encode_fn)
        
        return label_encode

    def labels(self):
        return self.valid_labels
        

    def __len__(self):
        return len(self.aef_files)

    def __getitem__(self,idx):
        aef_file = self.aef_files[idx]
        label_file = self.fbfm40_files[idx]

        aef_arr = xar.Dataset.to_dataarray(xar.open_zarr(aef_file,consolidated=False)).values[0]

        # fix array shapes
        # label_arr = xar.Dataset.to_dataarray(xar.open_zarr(label_file,consolidated=False)).values[0][0]
        label_arr = xar.Dataset.to_dataarray(xar.open_zarr(label_file,consolidated=False)).values[0]
        
        label_arr[np.isin(label_arr,np.array(self.ignore_index))] = -1
        label_arr = self.label_encode(label_arr)
        
        aef_tensor = torch.from_numpy(aef_arr.astype(np.float64)).float()
        label_tensor = torch.from_numpy(label_arr.astype(np.int64)).long()

        if self.transform:
            aef_tensor = self.transform(aef_tensor)


        output = {
            'image': {
                'aef': aef_tensor
            },
            'target': label_tensor
        }

        return output

class FM40Fuels30m(Dataset):
    FM40_LABELS = [
        91, 92, 93, 98, 99,
        101, 102, 103, 104, 105, 106, 107, 108, 109,
        121, 122, 123, 124,
        141, 142, 143, 144, 145, 146, 147, 148, 149,
        161, 162, 163, 164, 165,
        181, 182, 183, 184, 185, 186, 187, 188, 189,
        201, 202, 203, 204,
    ]


    def __init__(self,
        root_path='../data/pyrome_26/',
        im_size=160,
        ignore_index=[91,92,93,98,99],
        split='train',
        train_tiles = [
                            '00439', '00433', '00371', '00372', '00370', '00436', '00500', \
                            '00369', '00501', '00434', '00502', '00437', '00367', '00306', \
                            '00435', '00368', '00503', '00366', '00305', '00438', '00432'
                    ],
        transform = None
    ):
        
        self.root_path=root_path
        self.im_size = im_size
        self.ignore_index = ignore_index
        self.split = split
        self.train_tiles = train_tiles
        self.transform = transform

        tile_size=1600
        self.num_scenes = (tile_size // self.im_size)**2

        self.tilenums = glob.glob(os.path.join(root_path,'*'))
        self.valid_labels = [label for label in self.FM40_LABELS if not any(label == ignored for ignored in self.ignore_index)]
        self.label_encode = self.make_label_encoder()


        self.aef_files = sorted(glob.glob(os.path.join(root_path,'*','Scene_*','aef.zarr')))
        self.fbfm40_files = sorted(glob.glob(os.path.join(root_path,'*','Scene_*','fbfm40.zarr')))

        if self.split =='train':
            self.aef_files = [file for file in self.aef_files if any([tilenum in file for tilenum in self.train_tiles])]
            self.fbfm40_files = [file for file in self.fbfm40_files if any([tilenum in file for tilenum in self.train_tiles])]
        elif self.split =='test':
            self.aef_files = [file for file in self.aef_files if not any([tilenum in file for tilenum in self.train_tiles])]
            self.fbfm40_files = [file for file in self.fbfm40_files if not any([tilenum in file for tilenum in self.train_tiles])]

        

    def make_label_encoder(self):
        self.label_map = dict(zip(self.valid_labels,np.arange(len(self.valid_labels))))
        self.label_map[-1] = -1
        

        def encode_fn(x):
            return self.label_map[x]
        
        label_encode = np.vectorize(encode_fn)
        
        return label_encode

    def labels(self):
        return self.valid_labels
        

    def __len__(self):
        return len(self.aef_files)

    def __getitem__(self,idx):
        aef_file = self.aef_files[idx]
        label_file = self.fbfm40_files[idx]

        aef_arr = xar.Dataset.to_dataarray(xar.open_zarr(aef_file,consolidated=False)).values[0]

        # fix array shapes
        label_arr = xar.Dataset.to_dataarray(xar.open_zarr(label_file,consolidated=False)).values[0][0]
        # label_arr = xar.Dataset.to_dataarray(xar.open_zarr(label_file,consolidated=False)).values[0]
        
        label_arr[np.isin(label_arr,np.array(self.ignore_index))] = -1
        label_arr = self.label_encode(label_arr)
        
        aef_tensor = torch.from_numpy(aef_arr.astype(np.float64)).float()
        label_tensor = torch.from_numpy(label_arr.astype(np.int64)).long()

        if self.transform:
            aef_tensor = self.transform(aef_tensor)


        output = {
            'image': {
                'aef': aef_tensor
            },
            'target': label_tensor
        }

        return output