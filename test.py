import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import os
import matplotlib.pyplot as plt
from data import get_data
from monai.data import decollate_batch 
from utils import inference
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine
from monai.transforms import Compose, Invertd, Activationsd, AsDiscreted,EnsureTyped, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd
from utils import ConvertToMultiChannelBasedOnBratsClassesd


val_org_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ])

VAL_AMP = True

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")


if __name__ =="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=[8,16,32],
        strides=[2,2],
        norm=Norm.BATCH).to(device)
    
    root_dir = "/mnt/Enterprise2/shirshak/Task01_BrainTumour"
    
    train_loader, val_loader = get_data(root_dir)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
 

    checkpoint = torch.load("best_metric_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()  
    

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = inference(val_inputs, VAL_AMP, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}")

        