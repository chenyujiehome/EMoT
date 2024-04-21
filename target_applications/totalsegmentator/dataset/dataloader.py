import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch,DistributedSampler, pad_list_data_collate,list_data_collate
from torch.utils.data import random_split
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    SpatialPadd,
)
from monai.utils import set_determinism
import numpy as np
from sklearn.model_selection import KFold
import torch
from copy import deepcopy
class_map_part_prostate = {
    1: "PZ",
    2: "TZ",
    }
totalseg_taskmap_set = {
    'prostate': class_map_part_prostate,
}
class ScaleIntensity(MapTransform):

    def __init__(
        self,
        keys ,
        allow_missing_keys: bool = True,
        channel_wise: bool = True,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.channel_wise = channel_wise

    def scaler(self, img):
        if self.channel_wise:
            # Assume the channel dimension is the first dimension
            for i in range(img.shape[0]):
                if torch.max(img[i]) != 0:
                    img[i] = img[i] / torch.max(img[i])
        else:
            if torch.max(img) != 0:
                img = img / torch.max(img)
        return img

    def __call__(self, data) :
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
class NameData(MapTransform):


    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d['name'] = os.path.splitext(os.path.splitext(os.path.basename(d[key]))[0])[0]
        return d
def get_loader(args):
    set_determinism(seed=0)
    directory = args.dataset_path
    root_dir = tempfile.mkdtemp() if directory is None else directory
    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        NameData(keys=["image"]),
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
        RandSpatialCropd(keys=["image", "label"], roi_size=(args.roi_x, args.roi_y, args.roi_z), random_size=False),
        ScaleIntensity(keys=["image"],channel_wise=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
    val_transform = Compose(
    [
        NameData(keys=["image"]),
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        # SpatialPadd(keys=["image", "label"], spatial_size=[192, 192, 64]),
        # RandSpatialCropd(keys=["image", "label"], roi_size=[192, 192, 64], random_size=False),
        ScaleIntensity(keys=["image"],channel_wise=True),
    ]
)


    # ## Quickly load data with DecathlonDataset
    # 
    # Here we use `DecathlonDataset` to automatically download and extract the dataset.
    # It inherits MONAI `CacheDataset`, if you want to use less memory, you can set `cache_num=N` to cache N items for training and use the default args to cache all the items for validation, it depends on your memory size.


    # here we don't cache any data in case out of memory issue


    def train_val_split(train_dataset,test_dataset, k, i):

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        indices = range(len(train_dataset))
        
        for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
            if fold == i:
                train_subset_indices = train_indices
                val_subset_indices = val_indices
                break
        np.random.shuffle(train_subset_indices)
        np.random.shuffle(val_subset_indices)
        train_dataset.indices = train_subset_indices
        test_dataset.indices = val_subset_indices
        train_dataset.data=[train_dataset.data[i] for i in train_subset_indices]
        test_dataset.data=[test_dataset.data[i] for i in val_subset_indices]
        return train_dataset, test_dataset

    # 示例用法
    k = args.fold
    current_fold =args.fold_t

    # 假设root_dir, train_transform, val_transform已经定义
    train_dataset, val_dataset = train_val_split(
        train_dataset=DecathlonDataset(
            root_dir=root_dir,
            task="Task05_Prostate",
            transform=train_transform,  
            section="training",
            download=True,
            cache_rate=0.0,
            num_workers=4,
            val_frac=0.0,
        ),
        test_dataset=DecathlonDataset(
            root_dir=root_dir,
            task="Task05_Prostate",
            transform=val_transform,  
            section="training",
            download=False,
            cache_rate=0.0,
            num_workers=4,
            val_frac=0.0,
        ),
        k=k,
        i=current_fold,
    )
    test_dataset=DecathlonDataset(
            root_dir=root_dir,
            task="Task05_Prostate",
            transform=val_transform,  
            section="validation",
            download=False,
            cache_rate=0.0,
            num_workers=4,
            val_frac=0.0,
        )
    train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, collate_fn=list_data_collate,
                                 sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,collate_fn=list_data_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,collate_fn=list_data_collate)

    return train_loader, train_sampler, val_loader, test_loader
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/azureuser/prostate_data/")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--fold_t", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dist",  default=False,type=bool)
    parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    
    
    args=parser.parse_args()
    train_loader, train_sampler, val_loader, test_loader=get_loader(args)
