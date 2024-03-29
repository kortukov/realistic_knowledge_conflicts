import argparse
import munch
import yaml

import src.closedbook_exp as closedbook_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        config = munch.munchify(config)

    closedbook_exp.run_closed_book_experiment(config)
    





