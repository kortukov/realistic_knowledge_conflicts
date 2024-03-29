import collections
import re
import string

import src.model_utils as model_utils


def normalize_answer(answer_string):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer_string))))

def normalize_without_lowering(answer_string):
    """Remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer_string)))

def text_has_answer(answers, text) -> bool:
    """Check if any of the answers is in the text."""
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False


def compute_exact_match(prediction, ground_truth):
    """Check if normalized prediction is same as normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def get_generated_answer(outputs, tokenizer, input_len):
    """Get the generated answer from the model outputs."""

    generation_str = tokenizer.decode(
        outputs[0, input_len:].cpu(), skip_special_tokens=True
    )
    answer = generation_str.split("\n")[0]
    return answer


def check_if_model_output_is_correct_and_get_prediction(
    outputs, tokenizer, input_len, answers
):
    """Check if the model output is correct.

    Outputs is full model generation, including both the prompt and the
    generated text. Length of original input in tokens is given by input_len.
    """
    prediction = get_generated_answer(outputs, tokenizer, input_len)
    is_correct = any(
        [compute_exact_match(prediction, answer) for answer in answers]
    )
    return is_correct, prediction


def compute_f1(a_gold, a_pred, tokenizer):
    """Computes the token-level F1 score of two utterances, a_gold and a_pred.

    Code taken from SQuAD evaluation script: https://rajpurkar.github.io/SQuAD-explorer/
    """
    gold_toks = tokenizer.encode(a_gold)[1:] # Skip start of sequence token
    pred_toks = tokenizer.encode(a_pred)[1:]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1