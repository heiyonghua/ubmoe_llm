import os
import warnings


def prepare_for_catalogue(model_path,project_path,dataset_path):
    if not os.path.exists(model_path):
        warnings.warn(f"there is no folder in {model_path}")
    if not os.path.exists(project_path+"/output"):
        os.makedirs(project_path+"/output")
    if not os.path.exists(dataset_path):
        raise ValueError("the dataset path is wrong")
    return True