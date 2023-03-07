import importlib
from config import Config

# Name of the dataset class and the file 
# it resides in must have the same name
# For Now...

class ConfigLookUp():
    def __init__(self, config):
        assert config is not None
        self.config =  config

    def read_config()
        transform = find_class('transform', config.)
        dataset = 
        dataloader = 
        optimizer =
        model = 
        loss =   

def find_class(module_name, class_name):

    dir = model_name + '.' + class_name
    module_lib = importlib.import_module(dir)
    cl = getattr(module_lib, class_name)

    return cl
