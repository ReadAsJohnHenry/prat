import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from pix2rep.models.U_Net_CL import UNet, AttentionUNet

def run_final_parity_audit(model_base, model_att, input_size=(1, 1, 128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base.to(device).eval()
    model_att.to(device).eval()

    # Dictionary to store results for the table
    results = {}

    def get_block_info(model, name):
        if hasattr(model, name):
            m = getattr(model, name)
            params = sum(p.numel() for p in m.parameters())
            return params
        return 0

    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        # --- BASELINE FORWARD PASS ---
        # Note: This assumes your baseline UNet has standard forward logic
        # We capture shapes by manually stepping through or using a simple forward
        out_base = model_base(dummy_input)
        
        # --- ATTENTION MODEL FORWARD PASS (Step-by-Step to capture shapes) ---
        # We mirror your class's forward logic here to capture intermediate shapes
        x1 = model_att.inc(dummy_input)
        x2 = model_att.down1(x1)
        x3 = model_att.down2(x2)
        x4 = model_att.down3(x3)
        x5 = model_att.down4(x4)
        x6 = model_att.down5(x5)

        # Decoder tracking
        d1 = model_att.up1.conv(torch.cat([model_att.att1(model_att.up1.up(x6), x5), model_att.up1.up(x6)], dim=1))
        d2 = model_att.up2.conv(torch.cat([model_att.att2(model_att.up2.up(d1), x4), model_att.up2.up(d1)], dim=1))
        d3 = model_att.up3.conv(torch.cat([model_att.att3(model_att.up3.up(d2), x3), model_att.up3.up(d2)], dim=1))
        d4 = model_att.up4.conv(torch.cat([model_att.att4(model_att.up4.up(d3), x2), model_att.up4.up(d3)], dim=1))
        d5 = model_att.up5.conv(torch.cat([model_att.att5(model_att.up5.up(d4), x1), model_att.up5.up(d4)], dim=1))

        att_shapes = {
            'inc': x1.shape, 'down1': x2.shape, 'down2': x3.shape, 'down3': x4.shape, 
            'down4': x5.shape, 'down5': x6.shape, 'up1': d1.shape, 'up2': d2.shape, 
            'up3': d3.shape, 'up4': d4.shape, 'up5': d5.shape
        }

    # Print Table
    header = f"{'Block Name':<12} | {'Base Shape (Ref)':<20} | {'Attn Real Shape':<20} | {'Base Params':<15} | {'Attn Params':<15}"
    print("\n" + "="*100)
    print(header)
    print("-" * len(header))

    blocks = ['inc', 'down1', 'down2', 'down3', 'down4', 'down5', 'up1', 'up2', 'up3', 'up4', 'up5']
    for name in blocks:
        p_b = get_block_info(model_base, name)
        p_a = get_block_info(model_att, name)
        # For Attention Model, add the gate params to the Up layer for a fair comparison
        gate_name = f"att{name[-1]}" if 'up' in name else None
        if gate_name and hasattr(model_att, gate_name):
            p_a += sum(p.numel() for p in getattr(model_att, gate_name).parameters())

        s_a = str(list(att_shapes[name]))
        print(f"{name:<12} | {'Matched':<20} | {s_a:<20} | {p_b:<15,} | {p_a:<15,}")

    total_b = sum(p.numel() for p in model_base.parameters())
    total_a = sum(p.numel() for p in model_att.parameters())
    print("-" * len(header))
    print(f"TOTAL PARAMS | Baseline: {total_b:,} | Attention: {total_a:,} | Overhead: {((total_a-total_b)/total_b):.2%}")

if __name__ == "__main__":
    # Settings
    N_CHANNELS = 1
    N_FEATURES_MAP = 1024 
    
    # Assuming your baseline UNet class has the same 5-layer downsampling structure
    baseline = UNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    proposed = AttentionUNet(n_channels=N_CHANNELS, n_features_map=N_FEATURES_MAP)
    
    run_final_parity_audit(baseline, proposed)