# import clean_code.pix2rep.data as data
# import clean_code.pix2rep.CL_model as CL_model


# from clean_code.pix2rep.utils import Config

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
    training_loader_CL, validation_loader_CL = loaders_builder.build_loaders_for_CL_pretraining()

    fix_all_seeds(42)
    # Contrastive model
    cl_model = CL_model.CL_Model(cfg)

    # Run pre-training
    avg_train_losses, avg_val_losses = cl_model.run_training(training_loader_CL, validation_loader_CL)

    
    Xtr = [1, 2, 5, 10, 20, 50, 100]
    #Xtr = [1]
    for k in Xtr:
        for j in range(3):
        #for j in range(1):
            # limited_subjects_dic = limit_labeled_data(subjects_dic, k)
            
            fix_all_seeds(42)
            loaders_builder = data.Partially_Supervised_Loaders(dataset, all_slices, subjects_dic, cfg)
            loaders = loaders_builder.build_loaders(k)
            training_loader, validation_loader = loaders[0], loaders[1]

            # Contrastive model
            cl_model = CL_model.CL_Model(cfg)

            # Run finetuning
            if j == 0:
                avg_train_losses, avg_val_losses = cl_model.run_baseline_finetuning(training_loader, validation_loader, k)
            elif j == 1:
                cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)
                avg_train_losses, avg_val_losses = cl_model.run_finetuning(training_loader, validation_loader, k)
            elif j == 2:
                cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)
                avg_train_losses, avg_val_losses = cl_model.run_linear_probing(training_loader, validation_loader, k)
    
