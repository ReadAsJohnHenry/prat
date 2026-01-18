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
    # print(subjects_dic[0], all_slices[0])
    loaders_builder = data.Partially_Supervised_Loaders(dataset, all_slices, subjects_dic, cfg)
    test_volume_loader = loaders_builder.build_test_volume_loader()

    print("Baseline without fine-tuning")
    cl_model = CL_model.CL_Model(cfg)
    cl_model.load_backbone_model(cfg.contrastive_pretraining.save_path_backbone)
    test_losses, test_losses_detailed = cl_model.run_test_volume(test_volume_loader)
    print(f"Avg. Dice (ACDC): {test_losses_detailed}")