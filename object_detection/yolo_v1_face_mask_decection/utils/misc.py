from datetime import datetime



def generate_model_file_name(model_name, dataset_name, task, epochnum):
    # <model_name>_<dataset>_<task>_<details>_<epoch|date|version>
    # example: resnet50_cifar10_classification_lr0.001_epoch20
    
    date = datetime.now().strftime("%Y-%m-%d-%Hh_%Mm")
    return f"{model_name}_{dataset_name}_{task}_epoch{epochnum}_{date}"