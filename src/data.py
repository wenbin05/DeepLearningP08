"""Fixed official splits and deterministic subsets shared by every training seed."""
import hashlib
from datasets import load_dataset
from huggingface_hub import HfApi

DATASETS = {
    "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0", "article", "highlights"),
    "xsum": ("EdinburghNLP/xsum", "default", "document", "summary"),
}


def resolve_revision(repo, revision, repo_type="model"):
    # Always resolve floating refs to immutable Hub SHAs before reading artifacts.
    return HfApi().repo_info(repo, revision=revision, repo_type=repo_type).sha


def load_split(name, split, limit, data_seed, revision="main"):
    if limit is not None and limit <= 0:
        raise ValueError("Sample limits must be positive, or omitted for the full split")
    repo, subset, _, _ = DATASETS[name]
    sha = resolve_revision(repo, revision, "dataset")
    raw = load_dataset(repo, subset, split=split, revision=sha)
    size = len(raw)
    if limit is not None:
        if limit > size:
            raise ValueError(f"Requested {limit} samples from {split} of size {size}")
        raw = raw.shuffle(seed=data_seed).select(range(limit))
    # Hash stable example identifiers (or content if IDs are absent) in evaluation order.
    digest = hashlib.sha256()
    for row in raw:
        digest.update((str(row.get("id", row.get("document", row.get("article")))) + '\n').encode())
    return raw, {"dataset": name, "repository": repo, "config": subset,
                 "revision": sha, "split": split, "official_size": size,
                 "samples": len(raw), "data_seed": data_seed,
                 "ids_sha256": digest.hexdigest(), "fingerprint": raw._fingerprint}


def preprocess_function(examples, tokenizer, max_src=1024, max_tgt=128,
                        source="article", target="highlights"):
    inputs = tokenizer(examples[source], max_length=max_src, truncation=True)
    labels = tokenizer(text_target=examples[target], max_length=max_tgt, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs


def encode(raw, tokenizer, config, name):
    _, _, source, target = DATASETS[name]
    return raw.map(lambda ex: preprocess_function(
        ex, tokenizer, config.max_source_length, config.max_target_length, source, target),
        batched=True, remove_columns=raw.column_names)
