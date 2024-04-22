import argparse
import munch
from tqdm import tqdm
import yaml

import src.file_utils as file_utils
import src.evaluation.true as true


def divide_dataset_into_nli_true_false(data):
    print(f"Loading TRUE NLI classifier")
    nli_classifier = true.TrueNLIClassifier()
    print(f"Loaded NLI classifier on device {nli_classifier.model.device_map}")

    nli_true, nli_false = [], []
    for example in tqdm(data):
        question = example["question"]
        context = example["context"]
        answer = example["closedbook_answer"]
        pred_entail = nli_classifier.infer_entailment(context, question, answer)
        example["nli_pred"] = pred_entail
        if pred_entail:
            nli_true.append(example)
        else:
            nli_false.append(example)
    return nli_true, nli_false


def filter_examples_by_true(config):
    wrong_cb_path = config.wrong_examples_path.replace("<model_name>", config.model_name).replace("<dataset>", config.dataset)
    print(f"Loading wrong closed book examples from {wrong_cb_path}")
    wrong_cb_data = file_utils.load_parquet_dataset(wrong_cb_path)

    no_conflict_data, conflict_data = divide_dataset_into_nli_true_false(wrong_cb_data)

    conflict_path = config.conflict_examples_path.replace("<model_name>", config.model_name).replace("<dataset>", config.dataset)
    print(f"Saving examples with knowledge conflict to {conflict_path}")
    file_utils.save_dataset_to_parquet(conflict_data, conflict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        config = munch.munchify(config)

    filter_examples_by_true(config)







