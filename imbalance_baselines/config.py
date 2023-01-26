import os
from pathlib import Path

from . import DSET_NAMES, LOSS_NAMES, MODEL_NAMES, OPTIMIZER_NAMES, EVAL_NAMES, logger

import omegaconf.errors
from omegaconf import OmegaConf


class Config:
    def __init__(self, yaml_path, defaults_path=(Path(__file__).parent / "./default_config.yaml").resolve()):
        self.config = OmegaConf.load(yaml_path)
        self._defaults = OmegaConf.load(defaults_path)

        # Sanitize configuration
        assert self.config.Dataset.dataset_name in DSET_NAMES.keys()
        assert self.config.Training.optimizer.name in OPTIMIZER_NAMES.keys()
        for task in self.config.Training.tasks:
            assert task.model in MODEL_NAMES.keys()
            assert task.loss in LOSS_NAMES.keys()
        for eval_method in self.config.Evaluation:
            assert eval_method.method_name in EVAL_NAMES.keys()

        # Create directories if they do not exist
        if self.config.DataGeneration.plotting.draw_dataset_plots:
            os.makedirs(self.config.DataGeneration.plotting.plot_path, exist_ok=True)

        if self.config.Training.backup.save_models or self.config.Training.backup.load_models:
            os.makedirs(self.config.Training.backup.models_path, exist_ok=True)

        logger.info("Got configuration:")
        logger.info(OmegaConf.to_yaml(self.config))

    def __getitem__(self, item):
        try:
            return self.config[item]
        except omegaconf.errors.ConfigKeyError:  # Also captures errors arising from subsequent key accesses
            logger.warning(f'Value "{item}" is incomplete in configuration. Trying to use the default value...')
            retval = self._defaults[item]
            logger.warning(f'Got default value for {item}: {retval}')

            return retval

    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except omegaconf.errors.ConfigAttributeError:  # Also captures errors arising from subsequent attribute accesses
            logger.warning(f'Value "{item}" is incomplete in configuration. Trying to use the default value...')
            retval = getattr(self._defaults, item)
            logger.warning(f'Got default value for {item}: {retval}')

            return retval
