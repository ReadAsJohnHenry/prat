import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from torchinfo import summary
from pix2rep.models.U_Net_CL import UNet, AttentionUNet

def compare_layer_parameters(model_base, model_att):
    """
    Compares parameter counts for shared layers to ensure architectural parity.
    """
    print(f"{'Layer Name':<30} | {'Baseline Params':<20} | {'Attention Params':<20} | {'Status'}")
    print("-" * 85)
    
    # Dictionary of baseline modules for easy lookup
    base_dict = dict(model_base.named_modules())
    att_dict = dict(model_att.named_modules())
    
    # Focus on major blocks: Encoder (Down), Bottleneck, and Decoder (Up)
    # Note: Layer names might vary based on your specific implementation
    target_blocks = ['inc', 'down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4', 'outc']
    
    for block in target_blocks:
        if block in base_dict and block in att_dict:
            p_base = sum(p.numel() for p in base_dict[block].parameters())
            p_att = sum(p.numel() for p in att_dict[block].parameters())
            
            status = "MATCH" if p_base == p_att else "ADDITIONAL"
            print(f"{block:<30} | {p_base:<20} | {p_att:<20} | {status}")

def verify_architecture_equivalence():
    """
    Validates that the proposed AttentionUNet maintains the same 
    structural integrity as the baseline UNet.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (1, 128, 128) # Standard ACDC input patch size
    
    print("="*30 + " Baseline UNet Summary " + "="*30)
    baseline = UNet(n_channels=1, n_features_map=64).to(device)
    summary(baseline, input_size=input_size)
    
    print("\n" + "="*30 + " Proposed AttentionUNet Summary " + "="*30)
    att_model = AttentionUNet(n_channels=1, n_features_map=64).to(device)
    # This will display the output shape for every layer, 
    # verifying symmetry in skip connections and the bottleneck.
    summary(att_model, input_size=input_size)

    # Quantitative Parameter Analysis
    base_params = sum(p.numel() for p in baseline.parameters())
    att_params = sum(p.numel() for p in att_model.parameters())
    param_diff = att_params - base_params
    
    print("\n" + "-"*20 + " Fairness Metrics " + "-"*20)
    print(f"Total Parameter Increase: {param_diff}")
    print(f"Relative Overhead: {((att_params - base_params) / base_params)*100:.2f}%")
    print("Note: An overhead ~1% confirms that performance gains are due to "
          "feature selection, not increased model capacity.")

    compare_layer_parameters(baseline, att_model)

verify_architecture_equivalence()