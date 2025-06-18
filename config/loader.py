import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(Path(config_path), 'r') as file:
        return yaml.safe_load(file)
