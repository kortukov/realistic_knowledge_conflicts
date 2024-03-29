import json
import logging
import munch
import os
import pandas as pd
import pickle
import yaml

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


def print_args(args, output_dir=None, output_file=None):
    """Logs the args to the console and optionally to a file."""
    assert output_dir is None or output_file is None

    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")

    if output_dir is not None or output_file is not None:
        output_file = output_file or os.path.join(output_dir, "args.txt")
        with open(output_file, "w", encoding="utf-8") as file:
            for key, val in sorted(vars(args).items()):
                keystr = f"{key}" + (" " * (30 - len(key)))
                file.write(f"{keystr}   {val}\n")


def load_parquet_dataset(dataset_path: str):
    """Load a dataset from parquet to a list of dicts"""
    data = pd.read_parquet(dataset_path)
    return data.to_dict(orient="records")


def save_dataset_to_parquet(dataset, dataset_path: str):
    """Save a list of dicts to a parquet file"""
    print(f"Saving dataset of length {len(dataset)} to path {dataset_path}")
    # Create parent dirs if needed
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    dataframe = pd.DataFrame(dataset)
    dataframe.to_parquet(dataset_path)


def load_csv_dataset(dataset_path: str):
    """Load a dataset from csv to a list of dicts"""
    data = pd.read_csv(dataset_path)
    return data.to_dict(orient="records")


def save_dataset_to_csv(dataset, dataset_path: str):
    """Save a list of dicts to a csv file"""
    print(f"Saving dataset of length {len(dataset)} to path {dataset_path}")
    # Create parent dirs if needed
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    dataframe = pd.DataFrame(dataset)
    dataframe.to_csv(dataset_path)


def load_json_dataset(dataset_path: str):
    """Loads a dataset from a json file"""
    print("Loading dataset:", dataset_path)
    with open(dataset_path, encoding="utf-8") as file:
        return json.load(file)


def load_several_json_datasets(dataset_paths: "list[str]"):
    """Returns a dict mapping dataset names to datasets"""
    datasets = {}
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path).replace(".json", "")
        datasets[dataset_name] = load_json_dataset(dataset_path)
    return datasets


def save_json_dataset(dataset, dataset_path: str):
    """Saves a dataset to a json file

    Dataset must be json-serializable. It must be a list of dicts,
    where each dict is a datapoint.
    """
    print(f"Saving dataset of length {len(dataset)} to path {dataset_path}")
    # Create parent dirs if needed
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w', encoding="utf-8") as file:
        json.dump(dataset, file)


def save_to_pickle(object_to_pickle, path):
    """Saves an object to a pickle file"""
    # Create parent dirs if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(object_to_pickle, f)


def load_yaml_config_as_munch(config_path: str):
    """Load a yaml config file as a munch object.
    Munch is just a dict that allows you to access keys as attributes.
    """
    with open(config_path, 'r', encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)
        munch_config = munch.munchify(yaml_config)
    return munch_config


def dummy_logger(*args, **kwargs):
    """A dummy logger that does nothing"""
    pass

def replace_placeholders_in_paths(exp_config: munch.Munch, path_keys: list):
    """Replace placeholders <dataset_split>, <model_name> and <subset> in the 
    specified path_keys with actual values from exp_config.
    
    This is useful when we want to run the same experiment on different splits
    of the dataset, or with different models. 

    Mutates exp_config inplace.
    """

    if "dataset_split" in exp_config:
        for key in path_keys:
            if key not in exp_config:
                continue
            if exp_config[key] is None:
                continue

            exp_config[key] = exp_config[key].replace(
                "<dataset_split>", exp_config.dataset_split
            )


    if "model_name" in exp_config:
        for key in path_keys:
            if key not in exp_config:
                continue
            if exp_config[key] is None:
                continue

            exp_config[key] = exp_config[key].replace(
                "<model_name>", exp_config.model_name
            )

    if "subset" in exp_config:
        for key in path_keys:
            if key not in exp_config:
                continue
            if exp_config[key] is None:
                continue

            exp_config[key] = exp_config[key].replace(
                "<subset>", exp_config.subset
            )