
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

# def limit_labeled_data(subjects, k):

#     limited_subjects = {
#         "training": subjects["training"][:k],
#         "testing": subjects["testing"]
#     }

#     return limited_subjects

if __name__ == '__main__':

    cfg = Config().cfg

    # Dataloaders
    # dataset_folder = '/media/data/shared_data/ACDC/database'
    dataset_folder = 'ACDC/database'
    dataset = data.ACDC_dataset(dataset_folder)
    subjects_dic, all_slices = dataset.extract_and_preprocess_slices()
    # print(subjects_dic[0], all_slices[0])
    
    Xtr = [1, 2, 5, 10, 20, 50, 100]
    for k in Xtr:
        # limited_subjects_dic = limit_labeled_data(subjects_dic, k)

        loaders_builder = data.Partially_Supervised_Loaders(dataset, all_slices, subjects_dic, cfg)
        loaders = loaders_builder.build_loaders(k)
        training_loader, validation_loader = loaders[0], loaders[1]

        # Contrastive model
        cl_model = CL_model.CL_Model(cfg)
        cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)

        # Run pfinetuning
        avg_train_losses, avg_val_losses = cl_model.run_finetuning(training_loader, validation_loader, k)


