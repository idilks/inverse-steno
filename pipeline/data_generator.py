import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from tasks.base_task import BaseTask
from encoding_schemes.scheme_registry import SchemeRegistry
from agents.honest_agent import HonestAgent
from agents.steganographic_agent import SteganographicAgent
from agents.model_interface import BaseModel
from dataset.schema import DatasetRecord, DatasetMetadata


class DatasetGenerator:
    def __init__(
        self,
        task: BaseTask,
        model: BaseModel,
        model_name: str,
        scheme_registry: SchemeRegistry,
        temperature: float = 0.7,
        seed: int = 42,
        candidate_attempts: int = 3,
    ):
        self.task = task
        self.model = model
        self.model_name = model_name
        self.scheme_registry = scheme_registry
        self.temperature = temperature
        self.seed = seed
        self.rng = random.Random(seed)

        self.honest_agent = HonestAgent(
            model,
            model_name,
            temperature,
            max_retries=candidate_attempts,
        )
        self.stego_agents: Dict[str, SteganographicAgent] = {}
        for name, scheme in scheme_registry.all_schemes().items():
            self.stego_agents[name] = SteganographicAgent(
                honest_agent=self.honest_agent,
                encoding_scheme=scheme,
                seed=seed,
                candidate_attempts=candidate_attempts,
            )

        self._generation_stats = {}

    def generate_dataset(
        self,
        n_tasks: int = 100,
        bits_per_scheme: Optional[Dict[str, List[List[int]]]] = None,
    ) -> List[DatasetRecord]:
        task_instances = self.task.load_instances(n_tasks)

        # attach task object for q_value scoring
        for ti in task_instances:
            if ti.metadata is None:
                ti.metadata = {}
            ti.metadata["_task_obj"] = self

        self.rng.shuffle(task_instances)

        all_records: List[DatasetRecord] = []
        generation_stats = {
            "honest_generated": 0,
            "stego_generated": 0,
            "stego_round_trip_success": 0,
            "stego_round_trip_fail": 0,
            "task_correct_honest": 0,
            "task_correct_stego": 0,
        }

        total = len(task_instances)
        for task_idx, task_instance in enumerate(task_instances):
            if task_idx % max(1, total // 10) == 0:
                print(f"  Generating for task {task_idx + 1}/{total}...")

            sender_id = f"sender_{task_idx:05d}"

            honest_id = str(uuid.uuid4())
            honest_response = self.honest_agent.generate(task_instance)
            honest_correct = self.is_correct(honest_response.response_text, task_instance)
            honest_q = self.q_value(honest_response.response_text, task_instance)
            honest_wc = len(honest_response.response_text.split())

            honest_record = DatasetRecord(
                record_id=honest_id,
                synthetic_sender_id=sender_id,
                label=0,
                response_text=honest_response.response_text,
                task_question=task_instance.question,
                task_correct_answer=task_instance.correct_answer,
                task_answer_choices=task_instance.answer_choices,
                response_is_correct=honest_correct,
                q_value=honest_q,
                agent_type="honest",
                model_name=self.model_name,
                temperature=self.temperature,
                word_count=honest_wc,
                attempts=(honest_response.generation_metadata or {}).get("attempts"),
                task_metadata=self._clean_task_metadata(task_instance.metadata),
            )
            all_records.append(honest_record)
            generation_stats["honest_generated"] += 1
            if honest_correct:
                generation_stats["task_correct_honest"] += 1

            for scheme_name, stego_agent in self.stego_agents.items():
                scheme = stego_agent.encoding_scheme

                if bits_per_scheme and scheme_name in bits_per_scheme:
                    bit_list = bits_per_scheme[scheme_name]
                    target_bits = bit_list[task_idx % len(bit_list)]
                else:
                    target_bits = [
                        self.rng.randint(0, 1)
                        for _ in range(scheme.bits_per_message)
                    ]

                stego_id = str(uuid.uuid4())
                stego_response = stego_agent.generate(
                    task_instance,
                    target_bits=target_bits,
                )

                gen_meta = stego_response.generation_metadata or {}
                stego_correct = self.is_correct(stego_response.response_text, task_instance)
                round_trip = gen_meta.get("round_trip_success", False)
                stego_q = self.q_value(stego_response.response_text, task_instance)
                stego_wc = len(stego_response.response_text.split())

                stego_record = DatasetRecord(
                    record_id=stego_id,
                    synthetic_sender_id=sender_id,
                    label=1,
                    response_text=stego_response.response_text,
                    task_question=task_instance.question,
                    task_correct_answer=task_instance.correct_answer,
                    task_answer_choices=task_instance.answer_choices,
                    response_is_correct=stego_correct,
                    q_value=stego_q,
                    agent_type="steganographic",
                    model_name=self.model_name,
                    temperature=self.temperature,
                    scheme_name=scheme_name,
                    scheme_type=scheme.scheme_type,
                    target_bits=target_bits,
                    decoded_bits=gen_meta.get("decoded_bits"),
                    round_trip_success=round_trip,
                    paired_record_id=honest_id,
                    paired_honest_response=honest_response.response_text,
                    word_count=stego_wc,
                    honest_word_count=honest_wc,
                    word_count_diff=stego_wc - honest_wc,
                    attempts=(honest_response.generation_metadata or {}).get("attempts"),
                    candidate_attempts=gen_meta.get("candidate_attempts"),
                    encoding_metadata=gen_meta.get("encoding_metadata", {}),
                    task_metadata=self._clean_task_metadata(task_instance.metadata),
                )
                all_records.append(stego_record)

                generation_stats["stego_generated"] += 1
                if round_trip:
                    generation_stats["stego_round_trip_success"] += 1
                else:
                    generation_stats["stego_round_trip_fail"] += 1
                if stego_correct:
                    generation_stats["task_correct_stego"] += 1

        self._generation_stats = generation_stats

        print("\nGeneration complete:")
        print(f"  Honest: {generation_stats['honest_generated']}")
        print(f"  Stego: {generation_stats['stego_generated']}")
        rt_total = (
            generation_stats["stego_round_trip_success"]
            + generation_stats["stego_round_trip_fail"]
        )
        if rt_total > 0:
            print(
                f"  Round-trip success: "
                f"{generation_stats['stego_round_trip_success']}/{rt_total} "
                f"({generation_stats['stego_round_trip_success']/rt_total:.1%})"
            )

        return all_records

    # delegate task methods to actual task object
    def is_correct(self, response: str, task_instance):
        return self.task.is_correct(response, task_instance)

    def q_value(self, response: str, task_instance):
        return self.task.q_value(response, task_instance)

    def _clean_task_metadata(self, md):
        if not md:
            return {}
        out = dict(md)
        out.pop("_task_obj", None)
        return out

    def build_metadata(self, records: List[DatasetRecord]) -> DatasetMetadata:
        n_honest = sum(1 for r in records if r.label == 0)
        n_stego = sum(1 for r in records if r.label == 1)

        scheme_stats = {}
        for name in self.scheme_registry.all_schemes():
            scheme_records = [r for r in records if r.scheme_name == name]
            if scheme_records:
                rt_success = sum(1 for r in scheme_records if r.round_trip_success)
                correct = sum(1 for r in scheme_records if r.response_is_correct)
                mean_wc_diff = sum((r.word_count_diff or 0) for r in scheme_records) / len(scheme_records)
                mean_q = sum(r.q_value for r in scheme_records) / len(scheme_records)

                scheme_stats[name] = {
                    "total": len(scheme_records),
                    "round_trip_success_rate": rt_success / len(scheme_records),
                    "task_correct_rate": correct / len(scheme_records),
                    "mean_word_count_diff": mean_wc_diff,
                    "mean_q_value": mean_q,
                }

        return DatasetMetadata(
            dataset_name="inverse_planning_stego_agents_v2",
            description=(
                "Paired honest/steganographic responses for inverse-planning "
                "steganography detection experiments with richer attacker-side "
                "quality controls and metadata."
            ),
            generation_date=datetime.now().isoformat(),
            total_records=len(records),
            n_honest=n_honest,
            n_steganographic=n_stego,
            model_names=[self.model_name],
            task_source="AQUA-RAT or synthetic fallback",
            encoding_schemes=self.scheme_registry.summary(),
            generation_config={
                "temperature": self.temperature,
                "seed": self.seed,
            },
            statistics={
                "overall": self._generation_stats,
                "per_scheme": scheme_stats,
                "honest_task_correct_rate": (
                    self._generation_stats.get("task_correct_honest", 0)
                    / max(self._generation_stats.get("honest_generated", 1), 1)
                ),
                "stego_task_correct_rate": (
                    self._generation_stats.get("task_correct_stego", 0)
                    / max(self._generation_stats.get("stego_generated", 1), 1)
                ),
            },
        )