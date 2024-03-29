import transformers as tf
import torch

class BemMetric:
    """BEM metric from the paper 
    "Tomayto, Tomahto. Beyond Token-level Answer Equivalence for Question Answering Evaluation"
    """

    MODEL_URL = "kortukov/answer-equivalence-bem" 
    def __init__(self):
        self.tokenizer = tf.AutoTokenizer.from_pretrained(self.MODEL_URL)
        self.model = tf.AutoModelForSequenceClassification.from_pretrained(self.MODEL_URL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def tokenize_function(self, question, reference, candidate):
        text = f"[CLS] {candidate} [SEP]"
        text_pair = f"{reference} [SEP] {question} [SEP]"
        inputs = self.tokenizer(
            text=text,
            text_pair=text_pair,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return inputs.to(self.device)


    def compute_bem_score(self, question, reference, candidate):
        inputs = self.tokenize_function(question, reference, candidate)
        out = self.model(**inputs)

        bem_score = torch.nn.functional.softmax(out.logits, dim=-1)[0,1].item()
        return bem_score
    
    def correct_by_bem(self, question, reference, candidate):
        return self.compute_bem_score(question, reference, candidate) > 0.5

    def correct_by_disjunction_bem(self, question, reference, candidate):
        """Checks if candidate is equivalent to reference or reference is equivalent to candidate
        
        Answer equivalence is originally a asymmetric relation. Candidate is equivalent if it contains same or better information.
        In our setting we empirically found that disjuncted symmetric relation works better. 
        """
        return self.correct_by_bem(question, reference, candidate) or self.correct_by_bem(question, candidate, reference)

    def any_correct_by_bem(self, question, references, candidate):
        return any([self.correct_by_bem(question, reference, candidate) for reference in references])

    def any_correct_by_disjunction_bem(self, question, references, candidate):
        return any([self.correct_by_disjunction_bem(question, reference, candidate) for reference in references])
