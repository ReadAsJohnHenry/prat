import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import torch
import torch.nn as nn
from pix2rep.models.U_Net_CL import UNet, AttentionUNet

def run_dynamic_architecture_audit(model_base, model_att, input_size=(1, 1, 128, 128)):
    """
    Dynamically captures real output shapes and parameters using forward hooks.
    This ensures all layers (including Up-layers with skip connections) are verified.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base.to(device).eval()
    model_att.to(device).eval()

    # Containers to store captured metadata
    base_metadata = {}
    att_metadata = {}

    # Hook function to capture real-time output shapes
    def capture_shape_hook(layer_name, storage):
        def hook(module, input, output):
            # If the model returns a tuple (e.g., logits + att_maps), take the first element
            if isinstance(output, tuple):
                storage[layer_name] = list(output[0].shape)
            else:
                storage[layer_name] = list(output.shape)
        return hook

    # Register hooks for all functional blocks
    blocks = ['inc', 'down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4', 'outc']
    for name in blocks:
        if hasattr(model_base, name):
            getattr(model_base, name).register_forward_hook(capture_shape_hook(name, base_metadata))
        if hasattr(model_att, name):
            getattr(model_att, name).register_forward_hook(capture_shape_hook(name, att_metadata))

    # Execute a real forward pass with dummy data
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        _ = model_base(dummy_input)
        _ = model_att(dummy_input)

    # Printing the Comprehensive Comparison Table
    header = f"{'Block Name':<12} | {'Base Output Shape':<22} | {'Attn Output Shape':<22} | {'Base Params':<15} | {'Attn Params':<15}"
    print("\n" + "="*95)
    print("LAYER-BY-LAYER ARCHITECTURAL CONSISTENCY AUDIT")
    print("="*95)
    print(header)
    print("-" * len(header))

    for name in blocks:
        if name in base_metadata and name in att_metadata:
            m_b = getattr(model_base, name)
            m_a = getattr(model_att, name)
            
            # Real parameter counts
            p_b = sum(p.numel() for p in m_b.parameters())
            p_a = sum(p.numel() for p in m_a.parameters())
            
            print(f"{name:<12} | {str(base_metadata[name]):<22} | {str(att_metadata[name]):<22} | {p_b:<15,} | {p_a:<15,}")

    # Global Statistics
    total_b = sum(p.numel() for p in model_base.parameters())
    total_a = sum(p.numel() for p in model_att.parameters())
    
    print("-" * len(header))
    print(f"{'TOTAL MODEL':<12} | {'Identical Resolution':<22} | {'Identical Res.':<22} | {total_b:<15,} | {total_a:<15,}")
    print("="*95)
    print(f"REPORTABLE SUMMARY:")
    print(f" - Absolute Parameter Difference: {total_a - total_b:,}")
    print(f" - Relative Parameter Overhead:   {((total_a - total_b) / total_b):.2%}")
    print("="*95)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Settings based on your real training config
    N_CHANNELS = 1
    N_FEATURES_MAP = 1024 
    
    print(f"Auditing architectures with n_features_map={N_FEATURES_MAP}...")
    
    # Initialize real models
    baseline = UNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    proposed = AttentionUNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    
    # Run the dynamic audit
    run_dynamic_architecture_audit(baseline, proposed)