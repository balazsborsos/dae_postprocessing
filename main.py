from utils.parser import ConfigurationParser, parse_yaml_config

from train import train_model

if __name__ == "__main__":
    config_parser = ConfigurationParser()
    args = config_parser.parse_args()

    config = parse_yaml_config(args.config)

    mode = args.mode
    input_path = args.input

    # Use mode, config_path, and input_path in your script
    if mode == 'train':
        print("Training mode selected.")
        train_model(config, args.input)
    elif mode == 'inference':
        print("Inference mode selected.")
        raise NotImplementedError()
