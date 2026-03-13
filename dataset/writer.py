import json
import os
from dataclasses import asdict


class DatasetWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.paired_dir = os.path.join(output_dir, "paired")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.paired_dir, exist_ok=True)

    def write(self, records, metadata):
        dataset_path = os.path.join(self.output_dir, "dataset.jsonl")
        honest_path = os.path.join(self.output_dir, "honest_only.jsonl")
        stego_path = os.path.join(self.output_dir, "stego_only.jsonl")
        paired_path = os.path.join(self.paired_dir, "pairs.jsonl")
        meta_path = os.path.join(self.output_dir, "metadata.json")

        honest = [r for r in records if r.label == 0]
        stego = [r for r in records if r.label == 1]

        with open(dataset_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        with open(honest_path, "w", encoding="utf-8") as f:
            for r in honest:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        with open(stego_path, "w", encoding="utf-8") as f:
            for r in stego:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        with open(paired_path, "w", encoding="utf-8") as f:
            for r in stego:
                pair = {
                    "synthetic_sender_id": r.synthetic_sender_id,
                    "paired_record_id": r.paired_record_id,
                    "honest_response": r.paired_honest_response,
                    "stego_response": r.response_text,
                    "scheme_name": r.scheme_name,
                    "scheme_type": r.scheme_type,
                    "target_bits": r.target_bits,
                    "decoded_bits": r.decoded_bits,
                    "round_trip_success": r.round_trip_success,
                    "word_count_diff": r.word_count_diff,
                    "q_value": r.q_value,
                }
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)