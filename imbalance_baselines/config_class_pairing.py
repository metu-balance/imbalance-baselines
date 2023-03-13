# TODO: Remove this script file?
import importlib

def read_config(cfg):  # TODO temp reference func, may remove later
    dataset_class = find_module_component('dataset', cfg.Dataset.dataset_name)
    dataset = dataset_class(**cfg.Dataset.dataset_parameters)
    
    transform_class = find_module_component('transform', cfg.Transform.transform_name)
    transform = transform_class(**cfg.Transform.transform_parameters)
    
    dataloader_class = find_module_component('dataloader', cfg.Dataloader.dataloader_name)
    dataloader = dataloader_class(**cfg.Dataloader.dataloader_parameters)

    optimizer_class = find_module_component('optimizer', cfg.Optimizer.optimizer_name)
    optimizer = optimizer_class(**cfg.Optimizer.optimizer_parameters)

    model_class = find_module_component('model', cfg.Model.model_name)
    model = model_class(**cfg.Model.model_parameters)

    loss_class = find_module_component('loss', cfg.Loss.loss_name)
    loss = loss_class(**cfg.Loss.loss_parameters)   


# Name of the searched field and the file it resides in must have the same name (For Now...)
def find_module_component(module_name, component_name):  # TODO: may move to utils
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
