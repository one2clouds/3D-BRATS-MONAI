import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import os
import matplotlib.pyplot as plt



if __name__ =="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH).to(device)
    

    model.load_state_dict(torch.load(os.path.join("best_metric_model.pth")))
    model.eval()  

    with torch.no_grad():
        