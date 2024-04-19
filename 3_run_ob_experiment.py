import argparse
import munch
import yaml

import src.openbook_exp as openbook_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        config = munch.munchify(config)

    openbook_exp.run_openbook_experiment(config)
    





