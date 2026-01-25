import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from pix2rep.models.U_Net_CL import UNet, AttentionUNet

def check_architecture():
    dummy_input = torch.randn(1, 1, 128, 128).to("cuda:2")
    
    baseline = UNet(n_channels=1, n_features_map=64).to("cuda:2")
    att_model = AttentionUNet(n_channels=1, n_features_map=64).to("cuda:2")
    
    out_base = baseline(dummy_input)
    out_att = att_model(dummy_input)
    
    print(f"Baseline Output Shape: {out_base.shape}")
    print(f"Attention Output Shape: {out_att.shape}")
    
    # Optional: Count parameters to see the overhead of Attention Gates
    base_params = sum(p.numel() for p in baseline.parameters())
    att_params = sum(p.numel() for p in att_model.parameters())
    print(f"Parameter overhead: {((att_params - base_params) / base_params)*100:.2f}%")

check_architecture()