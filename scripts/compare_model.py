import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from pix2rep.models.U_Net_CL import UNet, AttentionUNet

import torch
import torch.nn as nn

def get_norm_layer(module):
    """Helper to find the normalization type used in a block."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            return m.__class__.__name__
    return "None"

def run_comprehensive_audit(model_base, model_att, input_size=(1, 1, 128, 128)):
    """
    Performs a real forward pass to extract layer-wise details.
    This replaces Yes/No with actual values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base.to(device).eval()
    model_att.to(device).eval()
    
    # Header
    header = f"{'Block Name':<12} | {'Output Shape':<22} | {'Base Params':<15} | {'Attn Params':<15} | {'Norm Method'}"
    print(header)
    print("-" * len(header))

    # To capture real shapes, we need to handle the forward pass carefully
    # We will use a dummy input and track its transformation through the encoder
    x_b = torch.randn(input_size).to(device)
    x_a = torch.randn(input_size).to(device)

    # List of encoder blocks (these can be run sequentially for shape extraction)
    encoder_blocks = ['inc', 'down1', 'down2', 'down3', 'down4']
    
    # Store intermediate shapes for the report
    for name in encoder_blocks:
        m_b = getattr(model_base, name)
        m_a = getattr(model_att, name)
        
        # Real Forward Pass to get Shape
        with torch.no_grad():
            x_b = m_b(x_b)
            x_a = m_a(x_a)
        
        # Get Parameter Counts
        p_b = sum(p.numel() for p in m_base.parameters())
        p_a = sum(p.numel() for p in m_att.parameters())
        
        # Get Norm Method
        norm = get_norm_layer(m_b)
        
        print(f"{name:<12} | {str(list(x_b.shape)):<22} | {p_base:<15,} | {p_attn:<15,} | {norm}")

    print("-" * len(header))
    print(f"{'TOTAL':<12} | {'[1, 1, 128, 128]':<22} | {sum(p.numel() for p in model_base.parameters()):<15,} | {sum(p.numel() for p in model_att.parameters()):<15,} | {'---'}")
    
    # Final Overhead Check
    total_b = sum(p.numel() for p in model_base.parameters())
    total_a = sum(p.numel() for p in model_att.parameters())
    print(f"\n[Summary] Total Parameter Overhead: {((total_a - total_b) / total_b):.2%}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # CONFIGURATION: n_features_map=1024 as per your setup
    N_CHANNELS = 1
    N_FEATURES_MAP = 1024 
    
    # Initialize models
    print(f"Initializing models with n_features_map={N_FEATURES_MAP}...")
    baseline = UNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    proposed = AttentionUNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    
    # Run the audit
    run_comprehensive_audit(baseline, proposed)