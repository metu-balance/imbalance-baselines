import importlib
import functools
from torchvision import transforms


# TODO: Handling None parameters is repetitive. Check if explosion (*list, **dict) can be used through a
#  controller function. Use config. parsing func from utils? Just modify get_partial_module?
class Registry:
    def __init__(self, cfg, static_transofrmations=True):
        """Partially initialize and store pipeline components."""
        # TODO: Leave partial / full initializations to the pipeline usage? Pipelines should be able to define multiple
        #  types of components (losses, models... etc.) together. Maybe registry should store the initialized objects
        #  types in lists...
        #  Let's consider this later, if we revive the idea of training tasks from v1.0.

        self.cfg = cfg

        if static_transofrmations:
            # Fully initialize transformations from the static parameters, return Compose
            self.training_transform_module = self.get_full_transforms(
                cfg.Transform.train_transform)
            self.testing_transform_module = self.get_full_transforms(
                cfg.Transform.test_transforms)
        else:
            # Partially initialize transformations, return a list
            self.training_transform_list = self.get_partial_transforms(
                cfg.Transform.train_transforms)
            self.testing_transform_list = self.get_partial_transforms(
                cfg.Transform.test_transforms)

        dataset_class = self.find_module_component(
            'dataset', cfg.Dataset.dataset_name)
        if cfg.Dataset.dataset_parameters != "None":
            self.partial_dataset_module = self.get_partial_module(
                dataset_class, cfg.Dataset.dataset_parameters)
        else:
            self.partial_dataset_module = dataset_class

        dataloader_class = self.find_module_component(
            'dataloader', cfg.Dataloader.dataloader_name)
        if cfg.Dataloader.dataloader_parameters != "None":
            self.partial_dataloader_module = self.get_partial_module(
                dataloader_class, cfg.Dataloader.dataloader_parameters)
        else:
            self.partial_dataloader_module = dataloader_class

        optimizer_class = self.find_module_component(
            'optimizer', cfg.Optimizer.optimizer_name)
        if cfg.Optimizer.optimizer_parameters != "None":
            self.partial_optimizer_module = self.get_partial_module(
                optimizer_class, cfg.Optimizer.optimizer_parameters)
        else:
            self.partial_optimizer_module = optimizer_class

        model_class = self.find_module_component('model', cfg.Model.model_name)
        if cfg.Model.model_parameters != "None":
            self.partial_model_module = self.get_partial_module(
                model_class, cfg.Model.model_parameters)
        else:
            self.partial_model_module = model_class

        loss_class = self.find_module_component('loss', cfg.Loss.loss_name)
        if cfg.Loss.loss_parameters != "None":
            self.partial_loss_module = self.get_partial_module(
                loss_class, cfg.Loss.loss_parameters)
        else:
            self.partial_loss_module = loss_class

    @staticmethod
    def get_partial_module(module, cfg_parameters):
        # TODO: Should keep previously created instances in a global sort of registry. For example,
        #   we need to ensure the same model instance is used for every model type used in a pipeline
        #   code.
        #   This behavior may be limited with the model objects only. At the least, we have to ensure
        #   that they are all initalized with the same model state.
        #   Maybe add a condition:
        #   if isinstance(module, nn.Module): <do state bookkeeping...>
        #   etc...
        # TODO: May also make the function static later

        def class_func(module, cfg_parameters, **kwargs):
            return module(**{**cfg_parameters, **kwargs})

        return functools.partial(class_func, module=module, cfg_parameters=cfg_parameters)

    # Name of the searched field and the file it resides in must have the same name (For Now...)
    # TODO: may move to utils or make static
    def find_module_component(self, module_name, component_name):
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

    def get_partial_transforms(self, transform_cfg_list):
        """Partially initialize the list of transformations described in the configuration, using the static parameters
        also given in the configuration.

        Complete initialization and composition into a single transforms.Compose object is the responsibility of
        the user.
        """

        transform_list = []

        for transform_config in transform_cfg_list:
            transform_name = transform_config.transform_name
            transform_parameters = transform_config.transform_parameters
            transform_class = self.find_module_component(
                'transform', transform_name)

            if transform_parameters != "None":
                partial_transform_module = self.get_partial_module(
                    transform_class, transform_parameters)

            transform_list.append(partial_transform_module)

        return transform_list

    def get_full_transforms(self, transform_cfg_list):
        """Compose and return a list of partially initialized transformations."""

        transform_list = []

        for transform_config in transform_cfg_list:
            transform_name = transform_config.transform_name
            transform_parameters = transform_config.transform_parameters
            transform_class = self.find_module_component(
                'transform', transform_name)

            if transform_parameters != "None":
                transform_module = transform_class(**transform_parameters)
            else:
                transform_module = transform_class()

            transform_list.append(transform_module)

        return transforms.Compose(transform_list)
