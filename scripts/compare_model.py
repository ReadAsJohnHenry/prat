import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from pix2rep.models.U_Net_CL import UNet, AttentionUNet

import torch
import torch.nn as nn

def get_norm_layer(module):
    """Find the normalization type used in a block."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            return m.__class__.__name__
    return "None"

def run_comprehensive_audit(model_base, model_att, input_size=(1, 1, 128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base.to(device).eval()
    model_att.to(device).eval()
    
    header = f"{'Block Name':<12} | {'Output Shape':<22} | {'Base Params':<15} | {'Attn Params':<15} | {'Norm Method'}"
    print(header)
    print("-" * len(header))

    # Real forward pass tracking
    x_b = torch.randn(input_size).to(device)
    x_a = torch.randn(input_size).to(device)

    # List of blocks to inspect
    blocks = ['inc', 'down1', 'down2', 'down3', 'down4']
    
    with torch.no_grad():
        for name in blocks:
            # Get the actual sub-modules from the models
            m_b = getattr(model_base, name)
            m_a = getattr(model_att, name)
            
            # Perform forward pass to get real shape
            x_b = m_b(x_b)
            x_a = m_a(x_a)
            
            # Calculate parameters for this specific block
            block_p_base = sum(p.numel() for p in m_b.parameters())
            block_p_attn = sum(p.numel() for p in m_a.parameters())
            
            # Identify normalization
            norm = get_norm_layer(m_b)
            
            # Print the row with real values
            print(f"{name:<12} | {str(list(x_b.shape)):<22} | {block_p_base:<15,} | {block_p_attn:<15,} | {norm}")

    # Final summary for the entire model
    total_b = sum(p.numel() for p in model_base.parameters())
    total_a = sum(p.numel() for p in model_att.parameters())
    
    print("-" * len(header))
    print(f"{'TOTAL MODEL':<12} | {'---':<22} | {total_b:<15,} | {total_a:<15,} | {'---'}")
    print(f"\n[Summary] Relative Parameter Overhead: {((total_a - total_b) / total_b):.2%}")

if __name__ == "__main__":
    N_CHANNELS = 1
    N_FEATURES_MAP = 1024 
    
    print(f"Initializing models with n_features_map={N_FEATURES_MAP}...")
    baseline = UNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    proposed = AttentionUNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    
    run_comprehensive_audit(baseline, proposed)