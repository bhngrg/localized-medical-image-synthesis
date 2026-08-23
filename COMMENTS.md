1) Register datasets: Allow CLI version, mention it in README


Files register_dataset.py, register_validation_dataset.py, build_h5_dataset.py, create_dataset_manifest.py modified:  all roots can be specified by CLI.
In all cases one can also provide --overwrite flag to ovewrite the existing results

python scripts/register_dataset.py \
    --data_root /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --yaml_dataset_path /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/dataset.yaml

python scripts/register_validation_dataset.py \
     --validation_data_root /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/ \
     --yaml_validation_dataset_path  /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/validation_dataset.yaml
     

 python scripts/build_h5_dataset.py \
    --yaml_dataset_path /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/dataset.yaml \
    --h5_root /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/H5_files

python scripts/create_dataset_manifest.py \
    --yaml_dataset_path /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/dataset.yaml \
    --h5_root /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/H5_files/ \
    --manifest_path /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/manifest.csv


2) What should I do after registering the dataset?

python scripts/train_patch_x0.py --h5-root /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/H5_files/ --manifest /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/manifest.csv

python scripts/synthesize_patch_x0.py --checkpoint ./checkpoints/full_train/final_patch_x0_diffusion_full_train.pt --h5-root /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/H5_files --manifest /Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/manifest.csv

