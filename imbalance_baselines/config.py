from pprint import pprint
from yaml import safe_load


class Config:
    def __init__(self, yaml_path, defaults_path="./default_config.yaml"):
        with open(yaml_path, "r") as f:
            self.config = safe_load(f)

        # Check if obligatory fields are present in config.
        conf_keys = self.config.keys()

        if "Dataset" not in conf_keys:
            raise Exception('"Dataset" field is missing from the configuration file.')
        else:
            # TODO: Check if a dataset name is given. If given, check whether the given dataset name
            #   is valid.
            pass
        
        if "DataGeneration" not in conf_keys:
            raise Exception('"DataGeneration" field is missing from the configuration file.')
        else:
            if "datasets_path" not in self.config["DataGeneration"].keys():
                raise Exception('"datasets_path" is missing from the DataGeneration field. Please'
                                ' provide a path to store the datasets in.')

        train_keys = self.config["Training"].keys()
        if "Training" in conf_keys:
            if "losses" in train_keys:
                # TODO: Check if given loss names are valid. Empty list is handled in next section.
                #for loss in self.config["Training"]["losses"]:
                #   if loss not in ...:  # Valid loss names list/set
                #     raise Exception("...")
                pass
            elif "backup" in train_keys:
                # If need to save/load models but no model path is specified, raise exception
                backup_keys = self.config["Training"]["backup"].keys()
                if (("save_models" in backup_keys
                        and self.config["Training"]["backup"]["save_models"])
                        or ("load_models" in backup_keys
                            and self.config["Training"]["backup"]["load_models"])) \
                        and "models_path" not in backup_keys:
                    raise Exception('Need model path to save/load models but "models_path" is not'
                                    ' specified in "backup" field under "Training".')
                    
        # Load default values if they do not exist in given YAML file. Warn user if default values are
        #   being used.
        with open(defaults_path, "r") as f:
            defaults = safe_load(f)
        
        # Datasets field and a valid name is known to exist now
        if self.config["Dataset"]["name"] == "IMB_CIFAR10" \
                and "cifar_imb_factor" not in self.config["Dataset"].keys():
            self.config["Dataset"]["cifar_imb_factor"] = defaults["Dataset"]["cifar_imb_factor"]
        
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
            for key in defaults["Training"].keys():
                if key not in train_keys:
                    self.config["Training"][key] = defaults["Training"][key]
            if not self.config["Training"]["losses"]:  # Empty list
                self.config["Training"]["losses"] = defaults["Training"]["losses"]
            # TODO: Check model config. too. (Or better, modify this check for the new composite
            #   config. type (loss, model, eval method...))
            
            # Repeat for sub-dictionaries
            for key in defaults["Training"]["backup"].keys():
                if key not in self.config["Training"]["backup"].keys():
                    self.config["Training"]["backup"][key] = defaults["Training"]["backup"][key]
                    
            for key in defaults["Training"]["printing"].keys():
                if key not in self.config["Training"]["printing"].keys():
                    self.config["Training"]["printing"][key] = defaults["Training"]["printing"][key]

        print("Got configuration:")
        pprint(self.config)
