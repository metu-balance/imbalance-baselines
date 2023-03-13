import os
from pathlib import Path

import omegaconf.errors
from omegaconf import OmegaConf
from . import logger

class Config:
    def __init__(self, yaml_path, defaults_path=(Path(__file__).parent / "./default_config.yaml").resolve()):
        self.config = OmegaConf.load(yaml_path)
        # TODO: handle defaults...
        #self._defaults = OmegaConf.load(defaults_path)

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
