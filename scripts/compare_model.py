import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
# Import from your model file
from pix2rep.models.U_Net_CL import UNet, AttentionUNet

def get_norm_layer(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            return m.__class__.__name__
    return "None"

def run_custom_audit(model_base, model_att, input_size=(1, 1, 128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base.to(device).eval()
    model_att.to(device).eval()

    base_shapes, att_shapes = {}, {}
    
    def get_hook(name, storage):
        def hook(module, input, output):
            storage[name] = list(output.shape) if not isinstance(output, tuple) else list(output[0].shape)
        return hook

    # --- MATCHING YOUR CLASS DEFINITIONS ---
    # Encoder blocks + Bottleneck (down5) + Decoder blocks
    blocks = ['inc', 'down1', 'down2', 'down3', 'down4', 'down5', 
              'up1', 'up2', 'up3', 'up4', 'up5']
    
    # Matching Attention Gates (Only exist in AttentionUNet)
    att_gates = ['att1', 'att2', 'att3', 'att4', 'att5']

    hooks = []
    for name in blocks:
        if hasattr(model_base, name):
            hooks.append(getattr(model_base, name).register_forward_hook(get_hook(name, base_shapes)))
        if hasattr(model_att, name):
            hooks.append(getattr(model_att, name).register_forward_hook(get_hook(name, att_shapes)))

    # Execute forward pass
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        _ = model_base(dummy_input)
        _ = model_att(dummy_input)

    # Print Table
    header = f"{'Block Name':<12} | {'Base Output Shape':<22} | {'Attn Output Shape':<22} | {'Base Params':<15} | {'Attn Params':<15}"
    print("\n" + "="*100)
    print(f"UNET ARCHITECTURE AUDIT (n_features_map={model_base.n_features_map})")
    print("="*100)
    print(header)
    print("-" * len(header))

    for name in blocks:
        # Get modules safely
        m_b = getattr(model_base, name, None)
        m_a = getattr(model_att, name, None)
        
        p_b = sum(p.numel() for p in m_b.parameters()) if m_b else 0
        p_a = sum(p.numel() for p in m_a.parameters()) if m_a else 0
        
        s_b = str(base_shapes.get(name, "N/A"))
        s_a = str(att_shapes.get(name, "N/A"))
        
        print(f"{name:<12} | {s_b:<22} | {s_a:<22} | {p_b:<15,} | {p_a:<15,}")

    # Add a special row for Attention Gates alone
    total_att_gate_p = sum(sum(p.numel() for p in getattr(model_att, g).parameters()) for g in att_gates)
    print("-" * len(header))
    print(f"{'Attn Gates':<12} | {'[Combined Overheads]':<22} | {'---':<22} | {'0':<15} | {total_att_gate_p:<15,}")
    
    # Overall
    total_b = sum(p.numel() for p in model_base.parameters())
    total_a = sum(p.numel() for p in model_att.parameters())
    print("-" * len(header))
    print(f"TOTAL MODEL  | Baseline: {total_b:,} | Attention: {total_a:,} | Overhead: {((total_a-total_b)/total_b):.2%}")
    print("="*100)
    
    for h in hooks: h.remove()

if __name__ == "__main__":
    # Settings
    N_CHANNELS = 1
    N_FEATURES_MAP = 1024 
    
    # Assuming your baseline UNet class has the same 5-layer downsampling structure
    baseline = UNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    proposed = AttentionUNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    
    run_custom_audit(baseline, proposed)