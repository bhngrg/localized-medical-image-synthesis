import kaggle
kaggle.api.authenticate()

kaggle.api.dataset_download_files(
    'awsaf49/brats20-dataset-training-validation', 
    path='/Volumes/Seagate_Backup_Disk/BIG_DATA/BraTS2020/', 
    unzip=True
)
