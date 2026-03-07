import argparse
import os

from agents.model_interface import load_model, StubModel
from encoding_schemes.scheme_registry import SchemeRegistry
from tasks.aqua_rat_task import AquaRatTask
from pipeline.data_generator import DatasetGenerator
from dataset.writer import DatasetWriter


def main():
    parser = argparse.ArgumentParser(description="Generate honest/steganographic datasets.")
    parser.add_argument("--model", type=str, default=os.getenv("DARTMOUTH_CHAT_MODEL", "stub"))
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="output/generated")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-attempts", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("INVERSE PLANNING AGENT DATA GENERATOR")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Tasks:       {args.n_tasks}")
    print(f"Output:      {args.output_dir}")
    print(f"Seed:        {args.seed}")
    print()

    if args.model == "stub":
        StubModel.reset_counter()

    task = AquaRatTask(dataset_path=args.dataset_path or None, seed=args.seed)
    scheme_registry = SchemeRegistry.default()
    model = load_model(args.model)

    generator = DatasetGenerator(
        task=task,
        model=model,
        model_name=args.model,
        scheme_registry=scheme_registry,
        temperature=args.temperature,
        seed=args.seed,
        candidate_attempts=args.candidate_attempts,
    )

    records = generator.generate_dataset(n_tasks=args.n_tasks)
    metadata = generator.build_metadata(records)

    writer = DatasetWriter(args.output_dir)
    writer.write(records, metadata)

    print("\nDataset written to:", os.path.abspath(args.output_dir))
    print("Files:")
    print("  - dataset.jsonl")
    print("  - honest_only.jsonl")
    print("  - stego_only.jsonl")
    print("  - paired/pairs.jsonl")
    print("  - metadata.json")


if __name__ == "__main__":
    main()