import src.file_utils as file_utils

ICL_SEPARATOR = ""

def format_icl_demonstration(icl_demo_prompt, icl_demonstration_example):
    question = icl_demonstration_example["question"]
    if not question.endswith("?"):
        question = question + "?"
    icl_demo_prompt = icl_demo_prompt.replace("<question>", question)
    icl_demo_prompt = icl_demo_prompt.replace("<context>", icl_demonstration_example["context"])

    cb_answer = icl_demonstration_example.get("closedbook_answer", "")
    if "<closedbook_answer>" in icl_demo_prompt and not cb_answer:
        print("WARNING: <closedbook_answer> in prompt but no closedbook_answer in example")
    icl_demo_prompt = icl_demo_prompt.replace("<closedbook_answer>", cb_answer)

    icl_demo_prompt = icl_demo_prompt.replace("<answer>", icl_demonstration_example["answers"][0])
    return icl_demo_prompt


def prepare_prompt_for_icl(
    original_prompt, icl_demo_prompt, icl_n, icl_dataset_path
):
    assert "<icl_demo>" in original_prompt
    icl_dataset = file_utils.load_parquet_dataset(icl_dataset_path)
    icl_demonstration_examples = icl_dataset[:icl_n]

    icl_demo_strings = [
        format_icl_demonstration(icl_demo_prompt, ex)
        for ex in icl_demonstration_examples
    ]
    full_icl_demonstration = ICL_SEPARATOR.join(icl_demo_strings)
    original_prompt = original_prompt.replace("<icl_demo>", full_icl_demonstration)
    return original_prompt


