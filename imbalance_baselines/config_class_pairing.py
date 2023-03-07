import importlib
from config import Config

# Name of the dataset class and the file 
# it resides in must have the same name
# For Now...



def read_config(cfg):
    
    dataset_class = find_class('dataset', cfg.Dataset.dataset_name)
    dataset = dataset_class(**cfg.Dataset.dataset_parameters)
    
    transform_class = find_class('transform', cfg.Transform.transform_name)
    transform = transform_class(**cfg.Transform.transform_parameters)
    
    dataloader_class = find_class('dataloader', cfg.Dataloader.dataloader_name)
    dataloader = dataloader_class(**cfg.Dataloader.dataloader_parameters)

    optimizer_class = find_class('optimizer', cfg.Optimizer.optimizer_name)
    optimizer = optimizer_class(**cfg.Optimizer.optimizer_parameters)

    model_class = find_class('model', cfg.Model.model_name)
    model = model_class(**cfg.Model.model_parameters)

    loss_class = find_class('loss', cfg.Loss.loss_name)
    loss = loss_class(**cfg.Loss.loss_parameters)   

def find_class(module_name, class_name):

    dir = module_name + '.' + class_name
    module_lib = importlib.import_module(dir)
    cl = getattr(module_lib, class_name)

    return cl
