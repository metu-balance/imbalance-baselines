import importlib
import functools
import config
from torchvision import transforms

class Registry:
    def __init__(self, cfg):
        self.cfg = cfg

    def read_config(self, cfg):  # TODO temp reference func, may remove later
        
        self.full_training_transform_module = get_full_transforms(cfg.Transform.train_transform)
        self.full_testing_transform_module = get_full_transforms(cfg.Transform.test_transforms)

        dataset_class = find_module_component('dataset', cfg.Dataset.dataset_name)
        self.partial_dataset_module = get_partial_module(dataset_class, cfg.Dataset.dataloader_parameters)

        dataloader_class = find_module_component('dataloader', cfg.Dataloader.dataloader_name)
        self.partial_dataloader_module = get_partial_module(dataloader_class, cfg.Dataloader.dataloader_parameters)

        optimizer_class = find_module_component('optimizer', cfg.Optimizer.optimizer_name)
        self.partial_optimizer_module = get_partial_module(optimizer_class, cfg.Optimizer.optimizer_parameters)

        model_class = find_module_component('model', cfg.Model.model_name)
        self.partial_model_module = get_partial_module(model_class, cfg.Model.model_parameters)
        
        loss_class = find_module_component('loss', cfg.Loss.loss_name)
        self.partial_loss_module = get_partial_module(loss_class, cfg.Loss.loss_parameters)   

    def get_partial_module(self, module, cfg_parameters):
        
        def class_func(module, cfg_parameters, **kwargs):
            return module(**{**cfg_parameters, **kwargs})

        return functools.partial(class_func, module = module, cfg_parameters = cfg_parameters)

    # Name of the searched field and the file it resides in must have the same name (For Now...)
    def find_module_component(self, module_name, component_name):  # TODO: may move to utils
        """Finds a specified field (variable, class or function) of a given sub-module of the library.

        :param module_name: Name of the sub-module, same as the name of the folder specifying it
        :param component_name: Name of the searched component. The searched component has to have the same name
            as the .py script in which it is defined.
        :return: The variable, function, or the class name. Note that returned functions are not called and classes are not
            instantiated.
         """

        module_dir = "imbalance_baselines." + module_name + "." + component_name
        module_lib = importlib.import_module(module_dir)
        cl = getattr(module_lib, component_name)

        return cl

    def get_full_transforms(self, transform_cfg_list):
        transform_list = []

        for transform_config in transform_cfg_list:
            transform_name = transform_config.transform_name
            transform_parameters = transform_config.transform_params

            transform_class = find_module_component('transform', transform_name)
            transform_module = transform_class(**transform_parameters)

        return transforms.Compose(transform_list)
