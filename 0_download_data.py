import argparse
import datasets
import os

DATA_DIR = "data"
 

def remove_tags_from_ctx(example):
    context = example['context']
    context = context.replace("[PAR]", "\n")

    context = context.replace("[DOC]", "")
    context = context.replace("[TLE]", "")
    context = context.replace("[SEP]", "")
    context = context.strip()
    example['context'] = context
    return example


def download_custom_dataset(custom_dataset_name, data_dir):
    full_data = datasets.load_dataset(custom_dataset_name)

    # For ICL we shuffle the original data and only save 10 examples.
    shuffled_data = full_data.shuffle(seed=42)
    print("We use 10 random examples for ICL dataset.")
    icl_data = shuffled_data.select(range(10))
    # For test data we use the remaining examples
    test_data = shuffled_data.select(range(10, len(shuffled_data)))

    icl_path = data_dir + f"/icl_{custom_dataset_name}.parquet"
    test_path = data_dir + f"/test_{custom_dataset_name}.parquet"

    icl_data.to_parquet(icl_path)
    test_data.to_parquet(test_path)


def download_data(args):
    data_dir = f"{DATA_DIR}/{args.dataset_type}"
    os.makedirs(data_dir, exist_ok=True)   

    if args.dataset_type == "custom":
        download_custom_dataset(args.custom_dataset_name, data_dir)
        return
    elif args.dataset_type == "test":
        # We use MrQA validation split as our test data
        dataset_split = "validation"
    else:
        # MrQA train split acts as ICL dataset in our experiments
        dataset_split = "train"
        
    print(f"Downloading {args.dataset_type} data.")
    full_data = datasets.load_dataset("mrqa", split=dataset_split)
    split_subset_names = list(set(full_data['subset']))

    for subset in split_subset_names: 
        print(f"Processing {subset} subset.")
        subset_data = full_data.filter(lambda ex: ex['subset'] == subset)

        if args.dataset_type == "icl":
            # For ICL we shuffle the original data and only save 10 examples.
            shuffled_data = subset_data.shuffle(seed=42)
            subset_data = shuffled_data.select(range(10))

        subset_data = subset_data.map(remove_tags_from_ctx)
        subset_split_path = data_dir + f"/{subset}.parquet"

        subset_data.to_parquet(subset_split_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-type", type=str, required=True, 
        choices=["test", "icl", "custom"], help="Which dataset to download."
    )

    parser.add_argument(
        "--custom-dataset-name", type=str, default=None,
        help="Huggingface hub id of the custom dataset."
    )

    args = parser.parse_args()

    download_data(args)

