from tqdm import tqdm

import src.evaluation.bem as bem
import src.evaluation.exact_match as exact_match
import src.evaluation.prompt_helpers as prompt_helpers
import src.model_utils as model_utils


def get_example_is_correct_fun(metric_name):
    """Get a function that checks if the model prediction is correct.
    
    Its parameters are question, context, answers, prediction. 
    """
    if metric_name == "EM":
        example_is_correct = lambda question, context, answers, prediction: any(
            [exact_match.compute_exact_match(prediction, answer) for answer in answers]
        )
    elif metric_name == "BEM":
        bem_metric = bem.BemMetric()
        example_is_correct = lambda question, context, answers, prediction: bem_metric.any_correct_by_disjunction_bem(
            question, answers, prediction
        )
    else:
        raise ValueError(f"Unknown metric name {metric_name}")
    return example_is_correct


class ClosedBookExampleResult:
    """Result of one example closed-book evaluation"""
    def __init__(self, example, correct, generated_answer, cb_in_ctx):
        self.example = example
        self.correct = correct
        self.generated_answer = generated_answer
        self.cb_in_ctx = cb_in_ctx

    def __repr__(self):
        return f"ClosedBookExampleResult(correct={self.correct})"

def evaluate_example_closed_book(
    model, tokenizer, example, custom_prompt, device, example_is_correct_fun
):
    """Evaluate a CausalLM model on a single example from the closed book
        dataset given a custom prompt.
    """
    answers = example["answers"]

    inputs, input_text = prompt_helpers.get_input_ids_with_prompt(
        tokenizer, example, custom_prompt, device
    )
    input_len = inputs.input_ids.shape[-1]

    outputs = model_utils.generate_answer(
        model, tokenizer, inputs, max_tokens_to_generate=10, device=device
    )

    generated_answer = exact_match.get_generated_answer(
        outputs, tokenizer, input_len
    )
    is_correct = example_is_correct_fun(example["question"], example["context"], answers, generated_answer)

    cb_in_ctx = prompt_helpers.tokenized_answer_found_in_model_inputs(
        answer=generated_answer, model_inputs=inputs, tokenizer=tokenizer,
    )

    return ClosedBookExampleResult(example, is_correct, generated_answer, cb_in_ctx)


def evaluate_closed_book(
    model, tokenizer, dataset, custom_prompt, device, metric_name
):
    """Evaluate a CausalLM model on the closed book dataset.
    Metric can be EM (Exact Match) or BEM (BERT Exact Match).

    Report the metric.
    Save the subset that is answered correctly. 
    Also save the subset that is answered wrongly.
    For both of them, save the generated answer.
    """
    num_correct = 0
    total_cost = 0.0
    correct_examples = []
    wrong_examples = []
    cb_in_ctx_correct = []
    cb_in_ctx_wrong = []

    example_is_correct_fun = get_example_is_correct_fun(metric_name)

    model.eval()
    for idx, example in tqdm(enumerate(dataset)):
        ex_result = evaluate_example_closed_book(
            model, tokenizer, example, custom_prompt, device, example_is_correct_fun
        )

        example["closedbook_answer"] = ex_result.generated_answer

        if ex_result.correct:
            num_correct += 1
            correct_examples.append(example)
            if ex_result.cb_in_ctx:
                cb_in_ctx_correct.append(example)
        else:
            wrong_examples.append(example)
            if ex_result.cb_in_ctx:
                cb_in_ctx_wrong.append(example)

    num_cb_in_ctx = len(cb_in_ctx_correct) + len(cb_in_ctx_wrong)

    additional_results = {
        "cb_in_ctx_ratio": num_cb_in_ctx / len(dataset),
        "incorrect_given_cb_in_ctx": len(cb_in_ctx_wrong) / num_cb_in_ctx,
    }
        

    correct_pct = num_correct / len(dataset)
    return correct_pct, correct_examples, wrong_examples, additional_results


class OpenBookExampleResult:
    """Result of one example open-book evaluation (with context)"""
    def __init__(self, example, correct, generated_answer, same_as_closed_book, cb_in_ctx, input_len):
        self.example = example
        self.correct = correct
        self.generated_answer = generated_answer
        self.same_as_closed_book = same_as_closed_book
        self.cb_in_ctx = cb_in_ctx
        self.input_len = input_len

    def __repr__(self):
        return f"OpenBookExampleResult(correct={self.correct}, same_as_closed_book={self.same_as_closed_book}, cb_in_ctx={self.cb_in_ctx})"


def evaluate_example_openbook(
    model,
    tokenizer,
    example,
    custom_prompt,
    device,
    example_is_correct_fun,
    example_is_same_fun,
    masking_strategy : str = None,
) -> OpenBookExampleResult:
    """Evaluate one example open-book (with context)."""


    if "answers" in example:
        ctx_answers = example["answers"]
    else:
        raise ValueError(f"Example must contain the fields 'answers'")

    closed_book_answers = [example["closedbook_answer"]]

    inputs, input_text = prompt_helpers.get_input_ids_with_prompt(
        tokenizer, example, custom_prompt, device
    )
    input_len = inputs.input_ids.shape[-1]

    if masking_strategy:
        inputs, input_text = prompt_helpers.mask_cb_answer(tokenizer, inputs, example, masking_strategy)

    cb_in_ctx = prompt_helpers.tokenized_answer_found_in_model_inputs(
        answer=closed_book_answers[0], model_inputs=inputs, tokenizer=tokenizer,
    )

    outputs = model_utils.generate_answer(
        model, tokenizer, inputs, max_tokens_to_generate=10, device=device
    )

    generated_answer = exact_match.get_generated_answer(
        outputs, tokenizer, input_len
    )
    ctx_is_correct = example_is_correct_fun(example["question"], example["context"], ctx_answers, generated_answer)

    same_as_closed_book = example_is_same_fun(example["question"], example["context"], closed_book_answers, generated_answer)

    correct = ctx_is_correct

    return OpenBookExampleResult(example, correct, generated_answer, same_as_closed_book, cb_in_ctx, input_len)


def evaluate_openbook(
    model, 
    tokenizer,
    dataset, 
    custom_prompt, 
    device,
    metric_name, 
    sameness_metric,
    masking_strategy : str = None,
):
    """Evaluate a CausalLM model on an open-book question answering task (with contexts).

    Each example has a question, context, ground-truth answer and a closed-book answer (given by the model 
    in previous experiments).
    
    For each example check if model predicts the Correct updated answer, or if wrong if 
    it is same as closed-book or different.
    (Assumption is that we run on the Wrong-closed-book subset).
    """
    incorrect_update = []
    retain_parametric = []
    correct_update = []

    ctx_lengths = []

    example_is_correct_fun = get_example_is_correct_fun(metric_name)
    example_is_same_fun = get_example_is_correct_fun(sameness_metric)

    cb_in_ctx_incorrect_update, cb_in_ctx_retain_parametric, cb_in_ctx_correct_update = 0, 0, 0

    model.eval()
    for example in tqdm(dataset):
        ex_result = evaluate_example_openbook(
            model,
            tokenizer,
            example,
            custom_prompt,
            device,
            example_is_correct_fun,
            example_is_same_fun,
            masking_strategy,
        )
        example["openbook_answer"] = ex_result.generated_answer

        if ex_result.correct:
            correct_update.append(example)
            cb_in_ctx_correct_update += ex_result.cb_in_ctx
        else:
            if ex_result.same_as_closed_book: 
                retain_parametric.append(example)
                cb_in_ctx_retain_parametric += ex_result.cb_in_ctx
            else:
                incorrect_update.append(example)
                cb_in_ctx_incorrect_update += ex_result.cb_in_ctx

        ctx_lengths.append(ex_result.input_len)


    results = {
        "incorrect_update": incorrect_update,
        "retain_parametric": retain_parametric,
        "correct_update": correct_update,
        "cb_in_ctx": {
            "incorrect_update": cb_in_ctx_incorrect_update,
            "retain_parametric": cb_in_ctx_retain_parametric,
            "correct_update": cb_in_ctx_correct_update,
        },
        "cb_not_in_ctx": {
            "incorrect_update": len(incorrect_update) - cb_in_ctx_incorrect_update,
            "retain_parametric": len(retain_parametric) - cb_in_ctx_retain_parametric,
            "correct_update": len(correct_update) - cb_in_ctx_correct_update,
        },
        "ctx_len_min": min(ctx_lengths),
        "ctx_len_avg": sum(ctx_lengths) / len(ctx_lengths),
        "ctx_len_max": max(ctx_lengths),
    } 
    return results


