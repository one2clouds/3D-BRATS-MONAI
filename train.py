import os 

import tempfile
from data import get_data

import torch 
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.data import decollate_batch 


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN (NVIDIA CUDA Deep Neural Network librart) benchmark
torch.backends.cudnn.benchmark = True


# define inference method
def inference(input, VAL_AMP):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

def training_phase(model,loss_function,optimizer,lr_scheduler,dice_metric,dice_metric_batch, post_trans, max_epochs,device):
    val_interval = 1
    VAL_AMP = True,

    epoch_loss_values = []
    metric_values = []

    metric_values_TC = []
    metric_values_WT = []
    metric_values_ET = []

    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(max_epochs):
        epoch_loss = 0
        model.train()
        step = 0
        for batch_data in tqdm(train_loader, desc=f'Training Epoch {epoch}/{max_epochs}', unit='epoch'):
            step +=1
            inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        print(f"Epoch {epoch+1} Avg. train loss: {epoch_loss:.4f}")

        if (epoch +1) % val_interval ==0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc=f'Training Epoch {epoch}/{max_epochs}', unit='epoch'):
                    val_inputs, val_labels = (val_data["image"].to(device),val_data["label"].to(device))
                    val_outputs = inference(val_inputs, VAL_AMP)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()

                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()

                metric_values_TC.append(metric_tc)
                metric_values_WT.append(metric_wt)
                metric_values_ET.append(metric_et)

                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch+1
                    torch.save(
                        model.state_dict(), 
                        os.path.join("best_metric_model.pth")
                        )
                    print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                        f"\nbest mean dice: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )

if __name__ == "__main__":
    max_epochs = 300
    root_dir = "/mnt/Enterprise2/shirshak/Task01_BrainTumour"

    train_loader, val_loader = get_data(root_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=[8,16,32],
        strides=[2,2],
        norm=Norm.BATCH).to(device)
    
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    training_phase(model,loss_function,optimizer,lr_scheduler,dice_metric,dice_metric_batch, post_trans,max_epochs,device)





