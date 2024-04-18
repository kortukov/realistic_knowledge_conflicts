import time
import numpy as np
import time
import torch
import transformers as tf




def load_tokenizer(model_name):
    """Load tokenizer from model_name"""
    if "llama" in model_name or "Llama" in model_name:
        return tf.LlamaTokenizer.from_pretrained(model_name)
    return tf.AutoTokenizer.from_pretrained(model_name)


def load_model_and_tokenizer(
    model_name, model_parallelism=False, cache_dir=None, auth_token=None, quantized=False,
):
    """Load model and tokenizer from model_name"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()


    config = tf.AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token
    if quantized:
        quant_config = tf.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, # Save 0.4 bits per parameter
            bnb_4bit_quant_type="nf4", # Recommended by https://huggingface.co/blog/4bit-transformers-bitsandbytes
            bnb_4bit_compute_dtype=torch.bfloat16, # bfloat 8b range, 7b precision, for large numbers (source https://huggingface.co/blog/hf-bitsandbytes-integration)
        )
        model_args["quantization_config"] = quant_config

    if "gemma" in model_name:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    if "GPTQ" in model_name:
        # I use model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"
        model = tf.AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            revision="gptq-3bit-128g-actorder_True",
        ).to(device)
        tokenizer = tf.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    else:
        model = tf.AutoModelForCausalLM.from_pretrained(
            model_name, **model_args
        ).eval()
        if not model_parallelism and not quantized:
            model = model.to(device)
        tokenizer = load_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    if model_parallelism:
        print(f"Device map: {model.hf_device_map}")

    return model, tokenizer, config, device


def generate_one_token(model, inputs):
    """Generate one token from the model."""
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        next_token = out.logits[0, -1, :].argmax().item()
    return next_token


def generate_one_token_with_activations(model, inputs, activations_to_collect):
    """Generate one token and return the token and the hidden states (activations)"""
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
        next_token = out.logits[0, -1, :].argmax().item()
        if activations_to_collect == "last":
            aggregated_activations = (h[:, -1, :] for h in out.hidden_states )
        elif activations_to_collect == "mean":
            aggregated_activations = (h.mean(dim=1) for h in out.hidden_states)
        elif activations_to_collect == "max":
            aggregated_activations = (h.max(dim=1)[0] for h in out.hidden_states)
        else:
            raise ValueError(f"Unknown activations_to_collect parameter: {activations_to_collect}, should be one of 'last', 'mean', 'max'")
        stacked_activations = np.stack([h.cpu().detach().numpy() for h in aggregated_activations])
    return next_token, stacked_activations


def generate_answer(
    model, tokenizer, inputs, max_tokens_to_generate, device
):
    """Generate answer by one token at a time.

    Return the generated tokens.
    """
    
    for i in range(max_tokens_to_generate):
        next_token= generate_one_token(model, inputs)    

        if next_token == tokenizer.eos_token_id:
            break
        
        # Add the generated token to the input for the next iteration
        inputs.input_ids = torch.cat(
            [inputs.input_ids, torch.tensor([[next_token]], device=device)], dim=-1
        )
        # Add attention mask for the new token
        inputs.attention_mask = torch.cat(
            [inputs.attention_mask, torch.tensor([[1]], device=device)], dim=-1
        )

    # Now input ids contains both the prompt and the generated tokens
    outputs = inputs.input_ids
    return outputs
