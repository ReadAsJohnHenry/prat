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

if __name__ == '__main__':

    cfg = Config().cfg

    # Dataloaders
    # dataset_folder = '/media/data/shared_data/ACDC/database'
    dataset_folder = 'ACDC/database'
    dataset = data.ACDC_dataset(dataset_folder)
    subjects_dic, all_slices = dataset.extract_and_preprocess_slices()

    loaders_builder = data.Partially_Supervised_Loaders(dataset, all_slices, subjects_dic, cfg)
    training_loader_CL, validation_loader_CL = loaders_builder.build_loaders_for_CL_pretraining()

    # Contrastive model
    cl_model = CL_model.CL_Model(cfg)

    # Run pre-training
    avg_train_losses, avg_val_losses = cl_model.run_training(training_loader_CL, validation_loader_CL)
