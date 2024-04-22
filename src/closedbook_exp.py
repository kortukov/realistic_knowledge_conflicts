import json
import munch 
import os

import src.evaluation.icl as icl
import src.evaluation.question_answering as question_answering
import src.validation as validation 
import src.file_utils as file_utils
import src.model_utils as model_utils


def run_closed_book_experiment(exp_config: munch.Munch):
    """Stage 1: Closed-book answer gathering.

    We run an LLM on a dataset closed-book to probe for its parametric knowledge.
    We also save the answers to later determine when a knowledge conflict between 
    parametric and contextual knowledge occurs.
    """
    custom_prompt = exp_config.custom_prompt

    logging_dict = {}
    def log_metric(metric_name, value):
        logging_dict[metric_name] = value

    if "metric_name" not in exp_config or not exp_config.metric_name:
        exp_config.metric_name = "EM"
    
    assert exp_config.metric_name in ["EM", "BEM"]

    model, tokenizer, _, device = model_utils.load_model_and_tokenizer(
        model_name=exp_config.model_name,
        model_parallelism=exp_config.model_parallelism,
        quantized=exp_config.quantized,
    )

    file_utils.replace_placeholders_in_paths(
        exp_config, 
        path_keys=["dataset_path", "correct_examples_path", "wrong_examples_path", "full_examples_path", "icl_dataset_path", "output_path"]
    )

    dataset = file_utils.load_parquet_dataset(exp_config.dataset_path)

    if "dataset_length" in exp_config and exp_config.dataset_length:
        dataset = dataset[:exp_config.dataset_length]
        print(f"Using only {exp_config.dataset_length} examples from the dataset")

    validation.assert_fields_exist(
        dataset=dataset, 
        fields=["context", "question", "answers"],
    )
    validation.ensure_string_fields(
        dataset=dataset,
        fields=["context", "question", "contextual_answer"],
    )

    if "<icl_demo>" in custom_prompt:
        assert "icl_demo_prompt" in exp_config
        assert "icl_n" in exp_config
        assert "icl_dataset_path" in exp_config
        custom_prompt = icl.prepare_prompt_for_icl(
            custom_prompt, exp_config.icl_demo_prompt, exp_config.icl_n, exp_config.icl_dataset_path
        )

    correct_ratio, correct_examples, wrong_examples, additional_results = question_answering.evaluate_closed_book(
        model, tokenizer, dataset, custom_prompt, device, exp_config.metric_name
    )
    log_metric(exp_config.metric_name, correct_ratio)

    correct_pct = correct_ratio * 100

    log_metric("Full data", len(dataset))
    log_metric("Closed-book correct", f"{len(correct_examples)} ({correct_pct:.2f}%)")
    log_metric("Closed-book wrong", f"{len(wrong_examples)} ({100 - correct_pct:.2f}%)")

    log_metric("Parametric answer in context", additional_results["cb_in_ctx_ratio"])
    log_metric("Incorrect out of parametric in context", additional_results["incorrect_given_cb_in_ctx"])

    print(f"{exp_config.metric_name}: {correct_pct:.2f}%")
    print(f"{len(correct_examples)} / {len(dataset)} correct examples")

    if exp_config.correct_examples_path:
        file_utils.save_dataset_to_parquet(
            correct_examples, exp_config.correct_examples_path
        )

    if exp_config.wrong_examples_path:
        file_utils.save_dataset_to_parquet(
            wrong_examples, exp_config.wrong_examples_path
        )
    
    if exp_config.full_examples_path:
        file_utils.save_dataset_to_parquet(
            dataset, exp_config.full_examples_path
        )
    
    if exp_config.output_path:
        # Ensure parent dir exists
        os.makedirs(os.path.dirname(exp_config.output_path), exist_ok=True)

        # Pretty print the logging dict as json to the output path
        with open(exp_config.output_path, "w") as f:
            f.write(json.dumps(logging_dict, indent=4))
    else:
        print("Output path, not specified")
        print(f"Results: ")
        print(json.dumps(logging_dict, indent=4))


