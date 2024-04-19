import transformers as tf


class TrueNLIClassifier:
    """Entailment classifier from paper 
    
    "TRUE: Re-evaluating Factual Consistency Evaluation"


    """
    MODEL_URL = "google/t5_xxl_true_nli_mixture"

    def __init__(self):
        model_args = {}
        model_args["device_map"] = "auto"

        self.tokenizer = tf.T5Tokenizer.from_pretrained(self.MODEL_URL)
        self.model = tf.T5ForConditionalGeneration.from_pretrained(self.MODEL_URL, **model_args)


    @staticmethod
    def format_example_for_autoais(context, question, answer):
        premise = context
        hypothesis = f"The answer to the question '{question}' is '{answer}'"
        return f"premise: {premise} hypothesis: {hypothesis}"


    def infer_entailment(self, context, question, answer):
        """Runs inference for assessing AIS between a premise and hypothesis.

        Args:
            example: Dict with the example data.
            tokenizer: A huggingface tokenizer object.
            model: A huggingface model object.

        Returns:
            A string representing the model prediction.
        """
        input_text = self.format_example_for_autoais(context, question, answer)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        entailment =  True if result == "1" else False
        return entailment
