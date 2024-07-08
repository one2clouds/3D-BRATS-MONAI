import os 

import tempfile
from data import get_data

import torch 
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from tqdm import tqdm
from monai.data import decollate_batch 
from utils import inference
import matplotlib.pyplot as plt
import numpy as np



# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN (NVIDIA CUDA Deep Neural Network librart) benchmark
torch.backends.cudnn.benchmark = True




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

        
        if (epoch +1) % val_interval ==0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc=f'Val Epoch {epoch}/{max_epochs}', unit='epoch'):
                    val_inputs, val_labels = (val_data["image"].to(device),val_data["label"].to(device))
                    val_outputs = inference(val_inputs, VAL_AMP, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    
                    # print(np.unique(np.stack(val_outputs)))
                    # print(np.stack(val_outputs).shape) # (1, 3, 240, 240, 155)
                    # print(np.stack(val_labels).shape)# (1, 3, 240, 240, 155)

                    temp_val_out= torch.as_tensor(np.stack(val_outputs)).squeeze(dim=0).argmax(0)
                    temp_val_label = torch.as_tensor(np.stack(val_labels)).squeeze(dim=0).argmax(0)

                    dice_metric(y_pred=temp_val_out, y=temp_val_label)
                    dice_metric_batch(y_pred=temp_val_out, y=temp_val_label)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)

                # print(dice_metric_batch.aggregate()) # tensor([0.0106, 0.0196, 0.0186], device='cuda:0')

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

                    torch.save({
                        'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss':loss,
                        }, "best_metric_model.pth")
                    
                    print("saved new best metric model")
                    print(
                        f"Epoch {epoch+1} Avg. train loss: {epoch_loss:.4f}"
                        f"\n current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                        f"\nbest mean dice: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
    return epoch_loss_values, metric_values, metric_values_TC, metric_values_WT, metric_values_ET, val_interval

if __name__ == "__main__":

    # Because of RuntimeError: received 0 items of ancdata
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    max_epochs = 5
    root_dir = "/mnt/Enterprise2/shirshak/Task01_BrainTumour"

    train_loader, val_loader,_ ,_, _, _ = get_data(root_dir)

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

    epoch_loss_values, metric_values, metric_values_TC, metric_values_WT, metric_values_ET, val_interval = training_phase(model,loss_function,optimizer,lr_scheduler,dice_metric,dice_metric_batch, post_trans,max_epochs,device)

    # Plot figures
    fig, ax = plt.subplots(2)

    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    ax[0].set_title("Epoch Average Loss")
    ax[0].plot(x, y, color="red")

    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    ax[1].set_title("Val Mean Dice")
    ax[1].plot(x, y, color="green")

    # To make some spacing between two plots
    fig.tight_layout()

    fig.savefig(f'plots/loss&dice.png')


    fig, ax = plt.subplots(3)

    plt.figure(figsize=(100,60))

    x = [val_interval * (i + 1) for i in range(len(metric_values_TC))]
    y = metric_values_TC
    ax[0].set_title("Val Mean Dice TC")
    ax[0].plot(x, y, color="blue")

    x = [val_interval * (i + 1) for i in range(len(metric_values_WT))]
    y = metric_values_WT
    ax[1].set_title("Val Mean Dice WT")
    ax[1].plot(x, y, color="brown")

    x = [val_interval * (i + 1) for i in range(len(metric_values_ET))]
    y = metric_values_ET
    ax[2].set_title("Val Mean Dice ET")
    ax[2].plot(x, y, color="purple")

    fig.tight_layout()
    fig.savefig(f'plots/dice_TC_WT_ET.png')

    # Check best model output with input image & label 



    checkpoint = torch.load("best_metric_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval() 
    with torch.no_grad():
        _,_,_,val_ds, _, _ = get_data(root_dir)
        val_input = val_ds[6]["image"].unsqueeze(0).to(device)
        roi_size = (128, 128, 64)
        sw_batch_size = 4
        VAL_AMP = True
        val_output = inference(val_input, VAL_AMP, model)
        val_output = post_trans(val_output[0])

        fig, ax = plt.subplots(1,4)
        for i in range(4):
            ax[i].set_title(f"image Channel {i}")
            ax[i].imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
        fig.tight_layout()
        fig.savefig(f'images/image_across_channel.png')

        fig, ax = plt.subplots(1,3)
        for i in range(3):
            ax[i].set_title(f"label channel {i}")
            ax[i].imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
        fig.tight_layout()
        fig.savefig(f'images/y_true_labels_across_channel.png')

        fig, ax = plt.subplots(1,3)
        for i in range(3):
            ax[i].set_title(f"output channel {i}")
            ax[i].imshow(val_output[i, :, :, 70].detach().cpu())
        fig.tight_layout()
        fig.savefig(f'images/y_pred_labels_across_channel.png')




    






