import importlib
import functools
from torchvision import transforms


class Registry:
    def __init__(self, cfg, static_transformations=True):
        """Partially initialize and store modules, using 'static' parameters defined in the configuration."""
        # TODO: static_transformations is a bit clumsy... Need to be able to determine this property automatically
        # TODO: Leave partial / full initializations to the pipeline usage? Pipelines should be able to define multiple
        #  types of components (losses, models... etc.) together. Maybe registry should store the initialized objects
        #  types in lists...
        #  Let's consider this later, if we revive the idea of training tasks from v1.0.

        self.cfg = cfg
        self.partial_modules = dict()

        if static_transformations:
            # Fully initialize transformations from the static parameters, return Compose
            self.training_transform_module = self.get_full_transforms(
                cfg.transform.train_transforms)
            self.testing_transform_module = self.get_full_transforms(
                cfg.transform.test_transforms)
        else:
            # Partially initialize transformations, return a list
            self.training_transform_list = self.get_partial_transforms(
                cfg.Transform.train_transforms)
            self.testing_transform_list = self.get_partial_transforms(
                cfg.Transform.test_transforms)

        # Create partial initializations for all modules other than transformations
        for module_type in cfg.keys():
            if module_type == "transform":
                # TODO: See the comment at the function's start. This is a bit clumsy.
                continue
            elif module_type in ["benchmark", "evaluation"]:
                continue  # TODO: Implement these and add to config. Skipping for now

            self.partial_modules[module_type] = self.get_partial_module(
                self.find_module_component(module_type, cfg[module_type].name),
                self.parse_module_parameters(cfg[module_type].parameters)
            )


    @staticmethod
    def parse_module_parameters(module_parameters):
        if module_parameters is None or module_parameters == "None":
            return dict()
        else:
            return module_parameters

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
        :return: The variable, function, or the class name. Note that returned functions are not called and classes are
            not instantiated.
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
            transform_name = transform_config.name
            transform_parameters = transform_config.parameters
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
            transform_name = transform_config.name
            transform_parameters = transform_config.parameters
            transform_class = self.find_module_component(
                'transform', transform_name)

            if transform_parameters != "None":
                transform_module = transform_class(**transform_parameters)
            else:
                transform_module = transform_class()

            transform_list.append(transform_module)

        return transforms.Compose(transform_list)
