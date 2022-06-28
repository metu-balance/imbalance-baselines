from pathlib import Path
from pprint import pprint
from yaml import safe_load
from . import DSET_NAMES, LOSS_NAMES, MODEL_NAMES, OPT_NAMES


class Config:
    def __init__(self, yaml_path, defaults_path=(Path(__file__).parent / "./default_config.yaml").resolve()):
        with open(yaml_path, "r") as f:
            self.config = safe_load(f)

        # Check if obligatory fields are present in config.
        conf_keys = self.config.keys()

        if "Dataset" not in conf_keys:
            raise Exception('"Dataset" field is missing from the configuration file.')
        else:
            if self.config["Dataset"]["name"] not in DSET_NAMES.keys():
                raise Exception("Unrecognized dataset name: " + self.config["Dataset"]["name"])
            
        if "DataGeneration" not in conf_keys:
            raise Exception('"DataGeneration" field is missing from the configuration file.')
        else:
            if "datasets_path" not in self.config["DataGeneration"].keys():
                raise Exception('"datasets_path" is missing from the DataGeneration field. Please'
                                ' provide a path to store the datasets in.')

        if "Training" in conf_keys:
            train_keys = self.config["Training"].keys()
            if "backup" in train_keys:
                # If need to save/load models but no model path is specified, raise exception
                backup_keys = self.config["Training"]["backup"].keys()
                if (("save_models" in backup_keys
                        and self.config["Training"]["backup"]["save_models"])
                        or ("load_models" in backup_keys
                            and self.config["Training"]["backup"]["load_models"])) \
                        and "models_path" not in backup_keys:
                    raise Exception('Need model path to save/load models but "models_path" is not'
                                    ' specified in "backup" field under "Training".')
            if "optimizer" in train_keys:
                if not ("name" in self.config["Training"]["optimizer"].keys()
                        and self.config["Training"]["optimizer"]["name"] in OPT_NAMES.keys()):
                    # Assign SGD as default name
                    self.config["Training"]["optimizer"]["name"] = "sgd"
            if "tasks" in train_keys:
                loss_names = LOSS_NAMES.keys()
                model_names = MODEL_NAMES.keys()
                for t in self.config["Training"]["tasks"]:
                   if t["loss"] not in loss_names:  # Valid loss names list/set
                     raise Exception("Unrecognized loss function name: " + t["loss"])
                   if t["model"] not in model_names:  # Valid model names list/set
                     raise Exception("Unrecognized model name: " + t["model"])
        
        
        # Load default values if they do not exist in given YAML file. Warn user if default values are
        #   being used.
        with open(defaults_path, "r") as f:
            defaults = safe_load(f)
        
        # Datasets field and a valid name is known to exist now
        if self.config["Dataset"]["name"] == "IMB_CIFAR10" \
                and "cifar10_imb_factor" not in self.config["Dataset"].keys():
            self.config["Dataset"]["cifar10_imb_factor"] = defaults["Dataset"]["cifar10_imb_factor"]
        
        # DataGeneration field and datasets_path are mandatory and known to exist.
        #   Get rest from defaults if they do not exist.
        datagen_keys = self.config["DataGeneration"].keys()
        for key in defaults["DataGeneration"].keys():
            if key not in datagen_keys:
                self.config["DataGeneration"][key] = defaults["DataGeneration"][key]

        # Ensure Training field and each Training key exists
        if "Training" not in conf_keys:
            self.config["Training"] = defaults["Training"]
        else:
            train_keys = self.config["Training"].keys()
            for key in defaults["Training"].keys():
                if key not in train_keys:
                    self.config["Training"][key] = defaults["Training"][key]
                    
            if not self.config["Training"]["tasks"]:  # Check for empty tasks list
                self.config["Training"]["tasks"] = defaults["Training"]["tasks"]
            
            # Repeat key checks for sub-dictionaries
            for key in defaults["Training"]["backup"].keys():
                if key not in self.config["Training"]["backup"].keys():
                    self.config["Training"]["backup"][key] = defaults["Training"]["backup"][key]
            
            for key in defaults["Training"]["optimizer"].keys():
                if key not in self.config["Training"]["optimizer"].keys():
                    self.config["Training"]["optimizer"][key] = defaults["Training"]["optimizer"][key]
            
            # If SGD is used, and the parameters are not given, load defaults
            if self.config["Training"]["optmizer"]["name"] == "sgd":
                for key in defaults["Training"]["optimizer"]["params"].keys():
                    if key not in self.config["Training"]["optimizer"].keys():
                        self.config["Training"]["backup"]["optimizer"][key] = \
                            defaults["Training"]["optimizer"]["params"][key]
            # TODO: Also set defaults for other optimizer types
                    
            for key in defaults["Training"]["printing"].keys():
                if key not in self.config["Training"]["printing"].keys():
                    self.config["Training"]["printing"][key] = defaults["Training"]["printing"][key]
            
            for key in defaults["Training"]["plotting"].keys():
                if key not in self.config["Training"]["plotting"].keys():
                    self.config["Training"]["plotting"][key] = defaults["Training"]["plotting"][key]

        print("Got configuration:")
        pprint(self.config)

    def __getitem__(self, item):
        return self.config[item]
