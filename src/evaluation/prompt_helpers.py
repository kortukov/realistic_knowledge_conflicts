import torch
import src.evaluation.exact_match as exact_match


def replace_placeholders_with_data(example, custom_prompt):
    """Replace placeholders in the custom prompt with actual data from the example."""
    # Prompt must contain placeholders for context and question
    # assert "<question>" in custom_prompt, "Prompt must contain <question>"

    context = example["context"]
    question = example["question"]
    closedbook_answer = example.get("closedbook_answer", "")
    answer = example.get("answers", "")[0]
    paraphrase_context = example.get("paraphrase_context", "")

    prompt = custom_prompt.replace("<context>", context)
    if not question.endswith("?"):
        question = question + "?"

    prompt = prompt.replace("<question>", question)
    prompt = prompt.replace("<closedbook_answer>", closedbook_answer)
    prompt = prompt.replace("<answer>", answer)
    prompt = prompt.replace("<paraphrase_context>", paraphrase_context)

    prompt = prompt.replace("<n>", "\n")
    prompt = prompt.replace("<inst>", "[INST]")
    prompt = prompt.replace("</inst>", "[/INST]")

    return prompt


def find_first_subtensor_index(tensor, sub_tensor):
    """Find the first index of sub_tensor in tensor."""
    for i in range(len(tensor) - len(sub_tensor) + 1):
        if (tensor[i : i + len(sub_tensor)] == sub_tensor).all():
            return i
    return -1


def tokenized_answer_found_in_model_inputs(answer, model_inputs, tokenizer):
    # We search for the answer as is, normalized and normalized without lowering
    all_answers = [
        answer,
        exact_match.normalize_answer(answer),
        exact_match.normalize_without_lowering(answer),
    ]

    all_answers_tokenized = [
        tokenizer(
            answer,
            return_tensors="pt",
        ).to(model_inputs.input_ids.device)
        for answer in all_answers
    ]
    all_answers_tensors = [
        answer_tokenized.input_ids[0][1:]
        for answer_tokenized in all_answers_tokenized
    ]

    text_tensor = model_inputs.input_ids[0]
    for answer_tensor in all_answers_tensors:
        sub_i = find_first_subtensor_index(text_tensor, answer_tensor)
        if sub_i != -1:
            return True

    return False



def get_masking_token(masking_strategy, tokenizer, example, device):
    if masking_strategy == "input_tokens_space":
        mask = torch.Tensor([259]).long().to(device) # 259 is the space token
    elif masking_strategy == "input_tokens_remove":
        mask = None
    elif masking_strategy == "input_tokens_unk":
        mask = torch.Tensor([tokenizer.unk_token_id]).long().to(device)
    elif masking_strategy == "input_tokens_entity":
        mask = torch.Tensor([7855]).long().to(device) # 7855 is the 'entity' token
    elif masking_strategy == "input_tokens_paraphrase_gpt":
        para_tokenized = tokenizer(
            exact_match.normalize_without_lowering(example["paraphrase_closedbook_answer_gpt"]),
            return_tensors="pt",
        ).to(device)
        mask = para_tokenized.input_ids[0][1:]
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}") 
    return mask


def mask_cb_answer_with_attention_mask(inputs, cb_inputs):
    prompt_inputs = inputs.input_ids[0]
    for i in range(len(prompt_inputs) - len(cb_inputs) + 1):
        if (prompt_inputs[i : i + len(cb_inputs)] == cb_inputs).all():
            inputs.attention_mask[0][i : i + len(cb_inputs)] = 0
    return inputs


def mask_input_tokens_of_cb_answer(tokenizer, inputs, cb_inputs, example, masking_strategy):
    device = inputs.input_ids.device

    cb_answer_found = True
    # Defense from infinite loop (in case of a bug)
    num_substitutions = 0
    max_substitutions = 50
    while cb_answer_found:
        prompt_inputs = inputs.input_ids[0]
        sub_i = find_first_subtensor_index(prompt_inputs, cb_inputs)
        if sub_i == -1:
            cb_answer_found = False
        else:
            # CB answer is found, masking it out using attention mask
            mask = get_masking_token(masking_strategy, tokenizer, example, device)

            unmasked_tensor = prompt_inputs
            # Replace all the tokens of CB answer with one token of the mask
            if mask is not None:
                new_tensor = torch.cat([unmasked_tensor[:sub_i], mask, unmasked_tensor[sub_i + len(cb_inputs):]]).to(device)
            else:
                new_tensor = torch.cat([unmasked_tensor[:sub_i], unmasked_tensor[sub_i + len(cb_inputs):]]).to(device)

            inputs.input_ids = new_tensor.unsqueeze(0)
            inputs.attention_mask = torch.ones_like(inputs.input_ids)

            num_substitutions += 1
            if num_substitutions > max_substitutions:
                break

    return inputs


def mask_a_tensor(tokenizer, inputs, example, masking_strategy, tensor_to_mask):

    split_strategy = masking_strategy.split("/")
    if len(split_strategy) == 1:
        masking_strategy, cb_answer_length = split_strategy[0], None
    elif len(split_strategy) == 2:
        masking_strategy, cb_answer_length = split_strategy
        cb_answer_length = int(cb_answer_length)
    else:
        raise ValueError(f"Invalid masking strategy: {masking_strategy}")

    if cb_answer_length:
        tensor_to_mask = tensor_to_mask[:cb_answer_length]

    if "input_text" in masking_strategy:
        input_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        text_to_mask = tokenizer.decode(tensor_to_mask, skip_special_tokens=True)
        if masking_strategy == "input_text_space":
            input_text = input_text.replace(text_to_mask, " ")
        elif masking_strategy == "input_text_paraphrase_gpt":
            para_text = exact_match.normalize_without_lowering(example["paraphrase_closedbook_answer_gpt"])
            input_text = input_text.replace(text_to_mask, para_text)
        else:
            raise ValueError(f"Unknown masking strategy: {masking_strategy}")       
        inputs = tokenizer(input_text, return_tensors="pt").to(inputs.input_ids.device)  

    elif masking_strategy == "attention_mask_full":
        inputs = mask_cb_answer_with_attention_mask(inputs, tensor_to_mask)
    elif "input_tokens" in masking_strategy:
        inputs = mask_input_tokens_of_cb_answer(tokenizer, inputs, tensor_to_mask, example, masking_strategy)
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")

    input_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

    return inputs, input_text


def mask_cb_answer(tokenizer, inputs, example, masking_strategy):
    """Find the CB answer in the input ids and mask it out using the specified strategy.
    
    Returns the modified inputs and decoded input text. 
    Looks for both the lowercased and unlowercased versions of the CB answer.
    """
    lowered_cb = exact_match.normalize_answer(example["closedbook_answer"])
    unlowered_cb = exact_match.normalize_without_lowering(example["closedbook_answer"])
    unnormalized_cb = example["closedbook_answer"]

    for normalized_cb_answer in [lowered_cb, unlowered_cb, unnormalized_cb]:
        cb_tokenized = tokenizer(
            normalized_cb_answer,
            return_tensors="pt",
        ).to(inputs.input_ids.device)
        cb_inputs = cb_tokenized.input_ids[0][1:]

        inputs, input_text = mask_a_tensor(tokenizer, inputs, example, masking_strategy, cb_inputs)
    return inputs, input_text


def get_input_ids_with_prompt(tokenizer, example, custom_prompt, device):
    """Create input ids from the example and the custom prompt.

    Replace <context> and <question> with the actual context and question. 
    Example is a dictionary that must contain the following keys:
    - context
    - question
    - closedbook_answer (optional)
    """
    prompt_with_data = replace_placeholders_with_data(example, custom_prompt) 

    inputs = tokenizer(prompt_with_data, return_tensors="pt")

    return inputs.to(device), prompt_with_data


def get_input_ids_with_doc_positions(tokenizer, example, custom_prompt, device):
    prompt = replace_placeholders_with_data(example, custom_prompt)

    sep1 =  "Context: " 
    sep2 = " Question:"
    assert sep1 in prompt
    assert sep2 in prompt
    prefix, docs = prompt.split(sep1)
    prefix = prefix + sep1
    docs, suffix = docs.split(sep2)
    suffix = sep2 + suffix

    prefix_input_ids = tokenizer(prefix, return_tensors="pt").input_ids
    # Remove the first token, which is the <s> token
    docs_input_ids = tokenizer(docs, return_tensors="pt").input_ids[:, 1:]
    suffix_input_ids = tokenizer(suffix, return_tensors="pt").input_ids[:, 1:]

    combined_input_ids = torch.cat([prefix_input_ids, docs_input_ids, suffix_input_ids], dim=1)
    docs_start = prefix_input_ids.shape[1]
    docs_end = docs_start + docs_input_ids.shape[1]

    assert all(combined_input_ids[0, docs_start:docs_end] == docs_input_ids[0])
    new_prompt = tokenizer.decode(combined_input_ids[0], skip_special_tokens=True)

    return combined_input_ids.to(device), new_prompt, docs_start, docs_end