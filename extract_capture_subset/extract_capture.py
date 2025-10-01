import yaml
import argparse
from pathlib import Path
import os

from .scripts.extract_location import extract_location


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reduced capture")
    parser.add_argument("--config", default="./extract_capture_subset/config.yaml", type=Path, help="YAML configuration file")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    f.close()
    return config

def extract_capture(config):
    capture_dir = config['capture_dir']
    endpoint = config['capture_subset_dir']
    ext_percent = config['extraction_percentage']
    
    locations = sorted(os.listdir(capture_dir))
    for location in locations:
        if location in ['codabench']:
            continue
        location_dir = os.path.join(capture_dir, location)
        endpoint_dir = os.path.join(endpoint, location)
        
        extract_location(location_dir, endpoint_dir, ext_percent)

def main():
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    extract_capture(config)
    
if __name__ == '__main__':
    main()