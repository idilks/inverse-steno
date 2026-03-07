"""AQUA-RAT dataset loader.

Loads from HuggingFace datasets or from a local JSON cache.
"""

import json
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "cache" / "aqua_rat.json"


def load_aqua_rat(n: int | None = None, split: str = "test") -> list[dict]:
    """Load AQUA-RAT questions.

    Tries local cache first, then HuggingFace datasets.

    Args:
        n: number of questions to load (None = all)
        split: dataset split ("train", "test", "validation")

    Returns:
        list of {"task_id": str, "question": str, "choices": str, "answer": str}
    """
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        if split in data:
            items = data[split]
            return items[:n] if n else items

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("deepmind/aqua_rat", split=split)
    except Exception as e:
        raise RuntimeError(
            f"Could not load AQUA-RAT dataset. Install `datasets` package "
            f"or place cached data at {CACHE_PATH}. Error: {e}"
        )

    items = []
    for i, row in enumerate(ds):
        items.append({
            "task_id": f"aqua_{split}_{i}",
            "question": row["question"],
            "choices": row["options"],
            "answer": row["correct"],
        })

    # Cache for next time
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            existing = json.load(f)
    existing[split] = items
    with open(CACHE_PATH, "w") as f:
        json.dump(existing, f)

    return items[:n] if n else items


def format_question(item: dict) -> str:
    """Format an AQUA-RAT item as a prompt string."""
    choices = item["choices"]
    if isinstance(choices, list):
        choices_str = "\n".join(choices)
    else:
        choices_str = choices
    return f"{item['question']}\n\nAnswer choices:\n{choices_str}"
