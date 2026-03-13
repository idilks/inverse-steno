"""Pre-generate reference candidates for AQUA-RAT questions.

Designed for HPC: uses concurrent API calls (default 10 parallel).
Set DARTMOUTH_CONCURRENCY=20 to increase parallelism.

Generates k candidates per question and caches them to disk.
Subsequent runs skip already-cached questions.

Usage:
    python -m scripts.generate_aqua_candidates --n-questions 100 --k 100
    DARTMOUTH_CONCURRENCY=20 python -m scripts.generate_aqua_candidates --n-questions 100 --k 100
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.dataset import load_aqua_rat, format_question
from src.detector.sampling import generate_candidates, load_cached
from scripts.bridge_eval import make_model_fn


async def main():
    parser = argparse.ArgumentParser(description="Pre-generate AQUA-RAT reference candidates")
    parser.add_argument("--n-questions", type=int, default=100, help="Number of AQUA-RAT questions")
    parser.add_argument("--k", type=int, default=100, help="Candidates per question")
    parser.add_argument("--model", default="anthropic.claude-haiku-4-5-20251001")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print(f"Loading AQUA-RAT ({args.split}, n={args.n_questions})...")
    questions = load_aqua_rat(n=args.n_questions, split=args.split)
    print(f"Loaded {len(questions)} questions")

    model_fn = make_model_fn(args.model, dry_run=False)

    # Build prompts
    prompts = []
    for item in questions:
        prompt = format_question(item)
        prompt += "\nPlease solve this step by step and provide your final answer."
        prompts.append((item["task_id"], prompt))

    # Check what's already cached
    n_cached = 0
    n_needed = 0
    for task_id, prompt in prompts:
        cached = load_cached(prompt, args.model, 1.0, args.k)
        if cached is not None:
            n_cached += 1
        else:
            n_needed += 1

    print(f"Already cached: {n_cached}, need to generate: {n_needed}")
    if n_needed == 0:
        print("All candidates already cached. Nothing to do.")
        return

    total_calls = n_needed * args.k
    print(f"Total API calls needed: {total_calls}")

    t0 = time.time()
    done = 0
    for task_id, prompt in prompts:
        # Skip if cached
        cached = load_cached(prompt, args.model, 1.0, args.k)
        if cached is not None:
            continue

        done += 1
        print(f"\n[{done}/{n_needed}] {task_id}: {prompt[:80]}...")
        candidates = await generate_candidates(
            prompt, model_fn, k=args.k, model_name=args.model
        )
        elapsed = time.time() - t0
        rate = done / elapsed * 60 if elapsed > 0 else 0
        remaining = (n_needed - done) / rate if rate > 0 else 0
        print(f"  Got {len(candidates)} candidates. "
              f"({done}/{n_needed}, {rate:.1f} questions/min, ~{remaining:.0f} min remaining)")

    elapsed = time.time() - t0
    print(f"\nDone. Generated candidates for {n_needed} questions in {elapsed/60:.1f} minutes.")
    print(f"Cache directory: cache/samples/")


if __name__ == "__main__":
    asyncio.run(main())
