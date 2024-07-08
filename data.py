from utils import ConvertToMultiChannelBasedOnBratsClassesd
from monai.transforms import (Activations,Activationsd,AsDiscrete,AsDiscreted,Compose,Invertd,LoadImaged,NormalizeIntensityd,Orientationd,RandFlipd,RandScaleIntensityd,RandSpatialCropd,Spacingd,EnsureTyped,EnsureChannelFirstd,RandShiftIntensityd)
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import CacheDataset, DataLoader, Dataset
import torch 
import os
import glob
from monai.utils import set_determinism


train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),
        RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

class BTSegDataset(Dataset):
    def __init__(self, file_names, transform):
        self.file_names = file_names
        self.transform = transform

    def __getitem__(self, index):
        file_names = self.file_names[index]
        dataset = self.transform(file_names) 
        return dataset
    
    def __len__(self):
        return 1#len(self.file_names)


def get_data(root_dir):
    set_determinism(seed=12345)
    train_images = sorted(glob.glob(os.path.join(root_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(root_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    # print(train_files[0])
    # print(os.path.exists(train_files[0]['label']))

    # CacheDataset to accelerate training and validation process, it's 10x faster than the regular Dataset.

    # train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=1.0, num_workers=4)
    # train_ds = Dataset(data=train_files, transform=train_transform)
    
    train_ds = BTSegDataset(train_files, train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

    # val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transform)
    val_ds = BTSegDataset(val_files, val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_ds, val_ds, train_files, val_files
