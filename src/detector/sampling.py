"""Reference sample generation and caching."""

import json
import hashlib
from pathlib import Path


CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "samples"


def _cache_key(prompt: str, model: str, temperature: float, k: int) -> str:
    h = hashlib.sha256(f"{prompt}|{model}|{temperature}|{k}".encode()).hexdigest()[:16]
    return h


def load_cached(prompt: str, model: str, temperature: float, k: int) -> list[str] | None:
    """Load cached reference samples if they exist."""
    key = _cache_key(prompt, model, temperature, k)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data["responses"]
    return None


def save_cache(prompt: str, model: str, temperature: float, k: int, responses: list[str]):
    """Cache reference samples for reuse."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(prompt, model, temperature, k)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump({
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "k": k,
            "responses": responses,
        }, f)


async def generate_candidates(
    prompt: str,
    model_fn,
    k: int = 50,
    model_name: str = "unknown",
    temperature: float = 1.0,
    use_cache: bool = True,
) -> list[str]:
    """Generate K reference responses, using cache when available.

    Args:
        prompt: the task prompt
        model_fn: async callable(prompt, n) -> list[str]
        k: number of candidates
        model_name: for cache keying
        temperature: for cache keying
        use_cache: whether to use disk cache

    Returns:
        list of K response strings
    """
    if use_cache:
        cached = load_cached(prompt, model_name, temperature, k)
        if cached is not None:
            return cached

    responses = await model_fn(prompt, k)

    if use_cache:
        save_cache(prompt, model_name, temperature, k, responses)

    return responses
