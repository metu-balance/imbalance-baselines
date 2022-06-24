from yaml import safe_load


class Config:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            self.config = safe_load(f)
            
            # TODO: Load default values if they do not exist in YAML file. Warn user if default values are
            #   being used.
        
        print("got config:", self.config)
        
