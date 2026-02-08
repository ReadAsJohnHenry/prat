
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pix2rep.data as data
import pix2rep.CL_model as CL_model


from pix2rep.utils import Config

import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from torchvision import transforms

def fix_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_all_seeds(42)

def run_multi_sigma_test(model, image, model_name, k_val, sigmas=[0, 1, 2, 4]):
    save_dir = os.path.join("hanyu/test1", "attention_infer")
    model.model.eval()
    # batch = next(iter(loader))
    # image = batch['image'][0:1].to('cuda') # (1, 1, H, W)
    
    fig, axes = plt.subplots(len(sigmas), 3, figsize=(12, 4 * len(sigmas)))
    fig.suptitle(f"Robustness Test: {model_name} (Xtr={k_val})", fontsize=16)

    for i, s in enumerate(sigmas):
        if s == 0:
            input_img = image
        else:
            input_img = transforms.GaussianBlur(kernel_size=7, sigma=s)(image)
        
        with torch.no_grad():
            output = model.model(input_img, return_att=True)
            atts = model.model.last_attention_maps
            # pred = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()
            att_map = atts[-1][0].squeeze().cpu().numpy()
        
        display_img = input_img[0].squeeze().cpu().numpy()
        
        axes[i, 0].imshow(display_img, cmap='gray')
        axes[i, 0].set_ylabel(f"Sigma={s}", fontsize=12)
        
        axes[i, 1].imshow(display_img, cmap='gray')
        axes[i, 1].imshow(att_map, cmap='jet', alpha=0.5)
        
        # axes[i, 2].imshow(pred, cmap='gray')
        
        if i == 0:
            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("Attention Map")
            # axes[i, 2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"multi_sigma_{model_name}_k{k_val}.png"))
    plt.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    cfg = Config().cfg

    # Dataloaders
    # dataset_folder = '/media/data/shared_data/ACDC/database'
    dataset_folder = 'ACDC/database'
    dataset = data.ACDC_dataset(dataset_folder)
    subjects_dic, all_slices = dataset.extract_and_preprocess_slices()
    # print(subjects_dic[0], all_slices[0])
    loaders_builder = data.Partially_Supervised_Loaders(dataset, all_slices, subjects_dic, cfg)
    # test_volume_loader = loaders_builder.build_test_volume_loader()
    training_loader_CL, validation_loader_CL = loaders_builder.build_loaders_for_CL_pretraining()
    
    print("Locking a specific slice for consistent inference test...")
    fixed_batch = next(iter(validation_loader_CL))
    fixed_image = fixed_batch[0].clone().detach().to(device)
    fixed_image = fixed_image.squeeze(1)

    print("Pretrained")
    cl_model = CL_model.CL_Model(cfg)
    cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)
    # test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
    # print(f"Avg. Dice (ACDC): {1 - test_losses}")
    run_multi_sigma_test(cl_model, fixed_image, "pre-trained", 100)

    Xtr = [1, 2, 5, 10, 20, 50, 100]
    # Xtr = [1]
    for k in Xtr:
        # limited_subjects_dic = limit_labeled_data(subjects_dic, k)

        # Contrastive model
        print(f"Xtr = {k}")

        print("Baseline (U-Net)")
        cl_model = CL_model.CL_Model(cfg)
        cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone.split(".")[0]+f"_bl_{k}.pth")
        cl_model.load_outconv_model(cfg.contrastive_pretraining.save_path_outconv_layer.split(".")[0]+f"_bl_{k}.pth")
        # test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
        # print(f"Avg. Dice (ACDC): {1 - test_losses}")
        run_multi_sigma_test(cl_model, fixed_image, "Baseline", k)

        print("Proposed (only Linear-probing)")
        cl_model = CL_model.CL_Model(cfg)
        cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)
        cl_model.load_outconv_model(cfg.contrastive_pretraining.save_path_outconv_layer.split(".")[0]+f"_lp_{k}.pth")
        # test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
        # print(f"Avg. Dice (ACDC): {1 - test_losses}")
        run_multi_sigma_test(cl_model, fixed_image, "LinearProbing", k)

        print("Proposed")
        cl_model = CL_model.CL_Model(cfg)
        cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone.split(".")[0]+f"_ft_{k}.pth")
        cl_model.load_outconv_model(cfg.contrastive_pretraining.save_path_outconv_layer.split(".")[0]+f"_ft_{k}.pth")
        # test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
        # print(f"Avg. Dice (ACDC): {1 - test_losses}")
        run_multi_sigma_test(cl_model, fixed_image, "FineTuning", k)
