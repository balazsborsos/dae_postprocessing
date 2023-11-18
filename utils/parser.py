import yaml

import argparse
from pathlib import Path


class ConfigurationParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Script for training or evaluation with configuration.')

        # Argument to specify mode (train or evaluation)
        self.parser.add_argument('mode', choices=['train', 'evaluation'], help='Mode: train or evaluation')
        # Argument to specify the path to the configuration YAML file
        self.parser.add_argument('-t', '--config', type=str, required=True, help='Path to the configuration file')
        # Argument to specify the path to the data
        self.parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input data')

    def parse_args(self):
        return self.parser.parse_args()


def parse_yaml_config(file_path: Path) -> dict:
    with open(file_path, 'r') as config_file:
        config_data = yaml.load(config_file, Loader=yaml.FullLoader)
    return config_data
