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
    def __init__(self, example, correct, generated_answer, cost):
        self.example = example
        self.correct = correct
        self.generated_answer = generated_answer
        self.cost = cost

    def __repr__(self):
        return f"ClosedBookExampleResult(correct={self.correct})"

def evaluate_example_closed_book(
    model, tokenizer, example, custom_prompt, device, example_is_correct_fun
):
    """Evaluate a CausalLM model on a single example from the closed book
        dataset given a custom prompt.
    """
    if "answers" in example:
        answers = example["answers"]
    elif "contextual_answer" in example:
        answers = [example["contextual_answer"]]
    else:
        raise ValueError(f"Example must contain fields 'answers' or 'contextual_answer'")

    inputs, input_text = prompt_helpers.get_input_ids_with_prompt(
        tokenizer, example, custom_prompt, device
    )
    input_len = inputs.input_ids.shape[-1]

    outputs, _, cost = model_utils.generate_answer(
        model, tokenizer, inputs, max_tokens_to_generate=10, device=device
    )

    generated_answer = exact_match.get_generated_answer(
        outputs, tokenizer, input_len
    )
    is_correct = example_is_correct_fun(example["question"], example["context"], answers, generated_answer)

    return ClosedBookExampleResult(example, is_correct, generated_answer, cost)


def evaluate_closed_book(
    model, tokenizer, dataset, custom_prompt, device, log_wandb, metric_name
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
        else:
            wrong_examples.append(example)
        
        total_cost += ex_result.cost

        wandb_dict = {
            metric_name: num_correct / (idx + 1),
            "Total cost": total_cost,
        }
        log_wandb(wandb_dict, step=idx)

    correct_pct = num_correct / len(dataset)
    return correct_pct, correct_examples, wrong_examples


class OpenBookExampleResult:
    """Result of one example open-book evaluation (with context)"""
    def __init__(self, example, correct, f1_score, generated_answer, same_as_closed_book, f1_closed_book, activations, cb_in_ctx, cost, input_len):
        self.example = example
        self.correct = correct
        self.f1_score = f1_score
        self.generated_answer = generated_answer
        self.same_as_closed_book = same_as_closed_book
        self.f1_closed_book = f1_closed_book
        self.activations = activations
        self.cb_in_ctx = cb_in_ctx
        self.cost = cost
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
    collect_activations : str = None,
    masking_strategy : str = None,
    print_input : bool = False,
) -> OpenBookExampleResult:
    """Evaluate one example open-book (with context)

    3 possible outcomes:
    - Correct (contextual) - according to factual context
    - Wrong (same) - same as closed-book answer
    - Wrong (different) - different from closed-book answer

    Returns activations if collect_activations is one of {'last', 'mean', 'max'}.
    Counts how often is the closed-book answer present in the context.
    """


    if "answers" in example:
        ctx_answers = example["answers"]
    elif "contextual_answer" in example:
        ctx_answers = [example["contextual_answer"]]
    else:
        raise ValueError(f"Example must contain fields 'answers' or 'contextual_answer'")

    closed_book_answers = [example["closedbook_answer"]]

    inputs, input_text = prompt_helpers.get_input_ids_with_prompt(
        tokenizer, example, custom_prompt, device
    )
    input_len = inputs.input_ids.shape[-1]

    if print_input:
        print(f"Example input text:\n{input_text}")

    if masking_strategy:
        inputs, input_text = prompt_helpers.mask_cb_answer(tokenizer, inputs, example, masking_strategy)


    cb_in_ctx = prompt_helpers.tokenized_answer_found_in_model_inputs(
        answer=closed_book_answers[0], model_inputs=inputs, tokenizer=tokenizer,
    )

    outputs, activations, cost = model_utils.generate_answer(
        model, tokenizer, inputs, max_tokens_to_generate=10, device=device, collect_activations=collect_activations
    )

    generated_answer = exact_match.get_generated_answer(
        outputs, tokenizer, input_len
    )
    ctx_is_correct = example_is_correct_fun(example["question"], example["context"], ctx_answers, generated_answer)
    f1_score = exact_match.compute_f1(ctx_answers[0], generated_answer, tokenizer)

    same_as_closed_book = example_is_same_fun(example["question"], example["context"], closed_book_answers, generated_answer)
    f1_closed_book = exact_match.compute_f1(closed_book_answers[0], generated_answer, tokenizer)


    correct = ctx_is_correct

    return OpenBookExampleResult(example, correct, f1_score, generated_answer, same_as_closed_book, f1_closed_book, activations, cb_in_ctx, cost, input_len)


def evaluate_openbook(
    model, 
    tokenizer,
    dataset, 
    custom_prompt, 
    device,
    log_wandb, 
    collect_activations,
    metric_name, 
    sameness_metric,
    masking_strategy : str = None,
):
    """Evaluate a CausalLM model on an open-book question answering task (with contexts).

    Each example has a question, context, contextual answer and a closed-book answer (given by the model 
    in previous experiments).
    
    For each example check if model predicts the correct contextual answer, or if wrong if 
    it is same as closed-book or different.
    (Assumption is that we run on the Wrong-closed-book subset).
    """
    wrong_diff = []
    wrong_same = []
    correct_ctx = []

    activations_wrong_diff = []
    activations_wrong_same = []
    activations_correct_ctx = []

    ctx_lengths = []

    example_is_correct_fun = get_example_is_correct_fun(metric_name)
    example_is_same_fun = get_example_is_correct_fun(sameness_metric)

    cb_in_ctx_wrong_diff, cb_in_ctx_wrong_same, cb_in_ctx_correct_ctx = 0, 0, 0
    f1_score_total, f1_cb_total = 0, 0

    total_cost = 0.0
    model.eval()
    for idx, example in enumerate(dataset):
        ex_result = evaluate_example_openbook(
            model,
            tokenizer,
            example,
            custom_prompt,
            device,
            example_is_correct_fun,
            example_is_same_fun,
            collect_activations,
            masking_strategy,
            print_input=(idx == 0),
        )
        example["openbook_answer"] = ex_result.generated_answer

        if ex_result.correct:
            correct_ctx.append(example)
            activations_correct_ctx.append(ex_result.activations)
            cb_in_ctx_correct_ctx += ex_result.cb_in_ctx
        else:
            if ex_result.same_as_closed_book: 
                wrong_same.append(example)
                activations_wrong_same.append(ex_result.activations)
                cb_in_ctx_wrong_same += ex_result.cb_in_ctx
            else:
                wrong_diff.append(example)
                activations_wrong_diff.append(ex_result.activations)
                cb_in_ctx_wrong_diff += ex_result.cb_in_ctx

        f1_score_total += ex_result.f1_score
        f1_cb_total += ex_result.f1_closed_book

        ctx_lengths.append(ex_result.input_len)

        total_cost += ex_result.cost

        wandb_dict = {
            "Wrong Different": len(wrong_diff) / (idx + 1),
            "Wrong Same": len(wrong_same) / (idx + 1),
            "Correct Contextual": len(correct_ctx) / (idx + 1),
            "CB in CTX Wrong Different": cb_in_ctx_wrong_diff / (len(wrong_diff) or 1),
            "CB in CTX Wrong Same": cb_in_ctx_wrong_same / (len(wrong_same) or 1),
            "CB in CTX Correct Contextual": cb_in_ctx_correct_ctx / (len(wrong_same) or 1),
            "F1 Score": f1_score_total / (idx + 1),
            "F1 Score CB": f1_cb_total / (idx + 1),
            "Total cost": total_cost,
            "Ctx len": ex_result.input_len,
        }
        log_wandb(wandb_dict, step=idx)

    results = {
        "wrong_diff": wrong_diff,
        "wrong_same": wrong_same,
        "correct_ctx": correct_ctx,
        "activations": {
            "wrong_diff": activations_wrong_diff,
            "wrong_same": activations_wrong_same,
            "correct_ctx": activations_correct_ctx,
        },
        "cb_in_ctx": {
            "wrong_diff": cb_in_ctx_wrong_diff,
            "wrong_same": cb_in_ctx_wrong_same,
            "correct_ctx": cb_in_ctx_correct_ctx,
        },
        "cb_not_in_ctx": {
            "wrong_diff": len(wrong_diff) - cb_in_ctx_wrong_diff,
            "wrong_same": len(wrong_same) - cb_in_ctx_wrong_same,
            "correct_ctx": len(correct_ctx) - cb_in_ctx_correct_ctx,
        },
        "f1_score": f1_score_total / len(dataset),
        "f1_score_cb": f1_cb_total / len(dataset),
        "ctx_len_min": min(ctx_lengths),
        "ctx_len_avg": sum(ctx_lengths) / len(ctx_lengths),
        "ctx_len_max": max(ctx_lengths),
    } 
    return results


