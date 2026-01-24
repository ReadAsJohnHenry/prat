
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

if __name__ == '__main__':

    cfg = Config().cfg

    # Dataloaders
    # dataset_folder = '/media/data/shared_data/ACDC/database'
    dataset_folder = 'ACDC/database'
    dataset = data.ACDC_dataset(dataset_folder)
    subjects_dic, all_slices = dataset.extract_and_preprocess_slices()
    # print(subjects_dic[0], all_slices[0])
    loaders_builder = data.Partially_Supervised_Loaders(dataset, all_slices, subjects_dic, cfg)
    test_volume_loader = loaders_builder.build_test_volume_loader()
    

    print("Pretrained")
    cl_model = CL_model.CL_Model(cfg)
    cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)
    #test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
    print(f"Avg. Dice (ACDC): {1 - test_losses}")

    Xtr = [1, 2, 5, 10, 20, 50, 100]
    #Xtr = [1]
    for k in Xtr:
        # limited_subjects_dic = limit_labeled_data(subjects_dic, k)

        # Contrastive model
        print(f"Xtr = {k}")

        print("Baseline (U-Net)")
        cl_model = CL_model.CL_Model(cfg)
        cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone.split(".")[0]+f"_bl_{k}.pth")
        cl_model.load_outconv_model(cfg.contrastive_pretraining.save_path_outconv_layer.split(".")[0]+f"_bl_{k}.pth")
        test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
        print(f"Avg. Dice (ACDC): {1 - test_losses}")

        print("Proposed (only Linear-probing)")
        cl_model = CL_model.CL_Model(cfg)
        cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)
        cl_model.load_outconv_model(cfg.contrastive_pretraining.save_path_outconv_layer.split(".")[0]+f"_lp_{k}.pth")
        test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
        print(f"Avg. Dice (ACDC): {1 - test_losses}")

        print("Proposed")
        cl_model = CL_model.CL_Model(cfg)
        cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone.split(".")[0]+f"_ft_{k}.pth")
        cl_model.load_outconv_model(cfg.contrastive_pretraining.save_path_outconv_layer.split(".")[0]+f"_ft_{k}.pth")
        test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
        print(f"Avg. Dice (ACDC): {1 - test_losses}")
