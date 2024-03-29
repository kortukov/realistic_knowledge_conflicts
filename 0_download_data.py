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


def download_data(args):
    data_dir = f"{DATA_DIR}/{args.dataset_type}"
    os.makedirs(data_dir, exist_ok=True)   

    if args.dataset_type == "test":
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
        choices=["test", "icl"], help="Which dataset to download."
    )

    args = parser.parse_args()

    download_data(args)

