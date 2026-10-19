"""Official Hugging Face Evaluate ROUGE; failures propagate, never approximate."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def rouge_metric():
    import evaluate
    return evaluate.load("rouge", revision="1bc20e7e9df922e5dfb597c31a4ea8467a89caff")


def compute_rouge_metrics(predictions, references):
    if not predictions or len(predictions) != len(references):
        raise ValueError("ROUGE requires nonempty, equally sized predictions and references")
    # Macro-average example F1, avoiding bootstrap randomness in Evaluate's default aggregator.
    scores = rouge_metric().compute(
        predictions=predictions, references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True, use_aggregator=False)
    return {k: float(np.mean(v) * 100) for k, v in scores.items()}


def decode_outputs(tokenizer, predictions, labels):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    return (tokenizer.batch_decode(predictions, skip_special_tokens=True),
            tokenizer.batch_decode(labels, skip_special_tokens=True))


def trainer_metrics(tokenizer, references=None):
    def compute(eval_prediction):
        predictions, decoded_references = decode_outputs(tokenizer, *eval_prediction)
        return compute_rouge_metrics(predictions, references if references is not None else decoded_references)
    return compute


if __name__ == "__main__":
    print(compute_rouge_metrics(["the cat sat on the mat"], ["the cat sat on the mat"]))
