import json
import os
import munch

import src.evaluation.icl as icl
import src.evaluation.question_answering as question_answering
import src.analysis as analysis
import src.validation as validation 
import src.file_utils as file_utils
import src.model_utils as model_utils


def do_logging_based_on_results(log_metric: callable, results: dict):
    incorrect_update = results["incorrect_update"] 
    retain_parametric = results["retain_parametric"]
    correct_update = results["correct_update"]
    num_examples = len(incorrect_update) + len(retain_parametric) + len(correct_update)

    incorrect_update_pct = len(incorrect_update) / num_examples
    retain_parametric_pct = len(retain_parametric) / num_examples
    correct_update_pct = len(correct_update) / num_examples

    log_metric("Incorrect update", incorrect_update_pct)
    log_metric("Retain parametric", retain_parametric_pct)
    log_metric("Correct update", correct_update_pct)

    log_metric("Num Incorrect update", len(incorrect_update))
    log_metric("Num Retain parametric", len(retain_parametric))
    log_metric("Num Correct update", len(correct_update))

    cb_in_ctx_incorrect_update_pct = results["cb_in_ctx"]["incorrect_update"] / (len(incorrect_update) or 1)
    cb_in_ctx_retain_parametric_pct = results["cb_in_ctx"]["retain_parametric"] / (len(retain_parametric) or 1)
    cb_in_ctx_correct_update_pct = results["cb_in_ctx"]["correct_update"] / (len(correct_update) or 1)

    log_metric("CB in Context Incorrect update", cb_in_ctx_incorrect_update_pct)
    log_metric("CB in Context Retain parametric", cb_in_ctx_retain_parametric_pct)
    log_metric("CB in Context Correct update", cb_in_ctx_correct_update_pct)

    num_cb_in_ctx = (
        results["cb_in_ctx"]["incorrect_update"] 
        + results["cb_in_ctx"]["retain_parametric"] 
        + results["cb_in_ctx"]["correct_update"]
    )
    p_cb_in_ctx = num_cb_in_ctx / num_examples
    log_metric("Overall CB in Context", p_cb_in_ctx)

    log_metric("Num CB in CTX Incorrect update", results["cb_in_ctx"]["incorrect_update"])
    log_metric("Num CB in CTX Retain parametric", results["cb_in_ctx"]["retain_parametric"])
    log_metric("Num CB in CTX Correct update", results["cb_in_ctx"]["correct_update"])

    log_metric("Num CB in Context", num_cb_in_ctx)

    num_not_cb_in_ctx = num_examples - num_cb_in_ctx
    num_not_cb_in_ctx_alt = (
        results["cb_not_in_ctx"]["incorrect_update"] 
        + results["cb_not_in_ctx"]["retain_parametric"] 
        + results["cb_not_in_ctx"]["correct_update"]
    )
    assert num_not_cb_in_ctx == num_not_cb_in_ctx_alt, "Calculations don't match" 


    p_not_cb_in_ctx = 1 - p_cb_in_ctx

    log_metric("Overall Not CB in Context", p_not_cb_in_ctx)

    log_metric("Num Not CB in CTX Incorrect update", results["cb_not_in_ctx"]["incorrect_update"])
    log_metric("Num Not CB in CTX Retain parametric", results["cb_not_in_ctx"]["retain_parametric"])
    log_metric("Num Not CB in CTX Correct update", results["cb_not_in_ctx"]["correct_update"])

    log_metric("Num Not CB in Context", num_not_cb_in_ctx)


    p_iu_given_cb_in_ctx = results["cb_in_ctx"]["incorrect_update"] / (num_cb_in_ctx or 1)
    log_metric("P(incorrect_update | cb_in_ctx)", p_iu_given_cb_in_ctx)

    p_iu_given_not_cb_in_ctx = results["cb_not_in_ctx"]["incorrect_update"] / (num_not_cb_in_ctx or 1)
    log_metric("P(incorrect_update | not cb_in_ctx)", p_iu_given_not_cb_in_ctx)

    # Test the hypothesis that p(wd) is different in the two groups
    p_value_iu = analysis.binomial_hypothesis_test(
        m_1=results["cb_in_ctx"]["incorrect_update"],
        n_1=num_cb_in_ctx,
        m_0=results["cb_not_in_ctx"]["incorrect_update"],
        n_0=num_not_cb_in_ctx,
    )
    log_metric("P-val IU", p_value_iu)
    
    p_rp_given_cb_in_ctx = results["cb_in_ctx"]["retain_parametric"] / (num_cb_in_ctx or 1)
    log_metric("P(retain_parametric | cb_in_ctx)", p_rp_given_cb_in_ctx)

    p_rp_given_not_cb_in_ctx = results["cb_not_in_ctx"]["retain_parametric"] / (num_not_cb_in_ctx or 1)
    log_metric("P(retain_parametric | not cb_in_ctx)", p_rp_given_not_cb_in_ctx)

    # Test the hypothesis that p(ws) is different in the two groups
    p_value_rp = analysis.binomial_hypothesis_test(
        m_1=results["cb_in_ctx"]["retain_parametric"],
        n_1=num_cb_in_ctx,
        m_0=results["cb_not_in_ctx"]["retain_parametric"],
        n_0=num_not_cb_in_ctx,
    )
    log_metric("P-val RP", p_value_rp)

    p_cu_given_cb_in_ctx = results["cb_in_ctx"]["correct_update"] / (num_cb_in_ctx or 1)
    log_metric("P(correct_update | cb_in_ctx)", p_cu_given_cb_in_ctx)

    p_cu_given_not_cb_in_ctx = results["cb_not_in_ctx"]["correct_update"] / (num_not_cb_in_ctx or 1)
    log_metric("P(correct_update | not cb_in_ctx)", p_cu_given_not_cb_in_ctx)

    # Test the hypothesis that p(cc) is different in the two groups
    p_value_cu = analysis.binomial_hypothesis_test(
        m_1=results["cb_in_ctx"]["correct_update"],
        n_1=num_cb_in_ctx,
        m_0=results["cb_not_in_ctx"]["correct_update"],
        n_0=num_not_cb_in_ctx,
    )
    log_metric("P-val CC", p_value_cu)


    # Memorization ratio = (ws / (ws + cc))
    memorization_ratio = retain_parametric_pct / (retain_parametric_pct + correct_update_pct)
    log_metric("Memorization Ratio", memorization_ratio)

    mem_ratio_given_cb_in_ctx = p_rp_given_cb_in_ctx / ((p_rp_given_cb_in_ctx + p_cu_given_cb_in_ctx) or 1)
    log_metric("Memorization Ratio | CB in Context", mem_ratio_given_cb_in_ctx)

    mem_ratio_given_not_cb_in_ctx = p_rp_given_not_cb_in_ctx / ((p_rp_given_not_cb_in_ctx + p_cu_given_not_cb_in_ctx) or 1)
    log_metric("Memorization Ratio | Not CB in Context", mem_ratio_given_not_cb_in_ctx)


    log_metric(f"Ctx len min", results["ctx_len_min"])
    log_metric(f"Ctx len avg", results["ctx_len_avg"])
    log_metric(f"Ctx len max", results["ctx_len_max"])

    print(f"Incorrect update, Percentage: {len(incorrect_update)} / {num_examples}, CB in Context: {cb_in_ctx_incorrect_update_pct}")
    print (f"Retain parametric, Percentage: {len(retain_parametric)} / {num_examples}, CB in Context: {cb_in_ctx_retain_parametric_pct}")
    print(f"Correct update, Percentage {len(correct_update)} / {num_examples}, CB in Context: {cb_in_ctx_correct_update_pct}")


def run_openbook_experiment(
    exp_config: munch.Munch,
):
    """Run open-book QA experiment on a dataset that contains "context", 
    "question", "answers" and "closedbook_answer" fields.
    """

    logging_dict = {}
    def log_metric(metric_name, value):
        logging_dict[metric_name] = value

    custom_prompt = exp_config.custom_prompt

    if "metric_name" not in exp_config or not exp_config.metric_name:
        exp_config.metric_name = "EM"
    
    assert exp_config.metric_name in ["EM", "BEM"]

    if "sameness_metric" not in exp_config or not exp_config.sameness_metric:
        exp_config.sameness_metric = exp_config.metric_name
    
    assert exp_config.sameness_metric in ["EM", "BEM"]

    file_utils.replace_placeholders_in_paths(
        exp_config, 
        path_keys=["dataset_path", "results_dir", "activation_dir", "icl_dataset_path", "output_path"]
    )

    model, tokenizer, _, device = model_utils.load_model_and_tokenizer(
        model_name=exp_config.model_name,
        model_parallelism=exp_config.model_parallelism,
        quantized=exp_config.quantized,
    )

    dataset = file_utils.load_parquet_dataset(exp_config.dataset_path)

    if "dataset_length" in exp_config and exp_config.dataset_length:
        dataset = dataset[:exp_config.dataset_length]
        print(f"Using only {exp_config.dataset_length} examples from the dataset")


    validation.assert_fields_exist(
        dataset=dataset, 
        fields=["context", "question", "answers", "closedbook_answer"],
    )
    validation.ensure_string_fields(
        dataset=dataset,
        fields=["context", "question", "closedbook_answer"],
    )


    if "masking_strategy" not in exp_config or not exp_config.masking_strategy:
        exp_config.masking_strategy = None

    if "<icl_demo>" in custom_prompt:
        assert "icl_demo_prompt" in exp_config
        assert "icl_n" in exp_config
        assert "icl_dataset_path" in exp_config
        custom_prompt = icl.prepare_prompt_for_icl(
            custom_prompt, exp_config.icl_demo_prompt, exp_config.icl_n, exp_config.icl_dataset_path
        )

    results = question_answering.evaluate_openbook(
        model,
        tokenizer,
        dataset,
        custom_prompt,
        device,
        exp_config.metric_name,
        exp_config.sameness_metric,
        exp_config.masking_strategy,
    )

    do_logging_based_on_results(log_metric, results)

    incorrect_update = results["incorrect_update"] 
    retain_parametric = results["retain_parametric"]
    correct_update = results["correct_update"]

    # Save examples in each category
    if exp_config.results_dir:
        file_utils.save_dataset_to_parquet(
            incorrect_update, exp_config.results_dir + "/incorrect_update.parquet"
        )
        file_utils.save_dataset_to_parquet(
            retain_parametric, exp_config.results_dir + "/retain_parametric.parquet"
        )
        file_utils.save_dataset_to_parquet(
            correct_update, exp_config.results_dir + "/correct_update.parquet"
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
