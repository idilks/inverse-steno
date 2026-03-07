from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class DatasetRecord:
    record_id: str
    synthetic_sender_id: str
    label: int
    response_text: str
    task_question: str
    task_correct_answer: str
    task_answer_choices: Optional[List[str]]
    response_is_correct: bool
    q_value: float
    agent_type: str
    model_name: str
    temperature: float

    scheme_name: Optional[str] = None
    scheme_type: Optional[str] = None
    target_bits: Optional[List[int]] = None
    decoded_bits: Optional[List[int]] = None
    round_trip_success: Optional[bool] = None

    paired_record_id: Optional[str] = None
    paired_honest_response: Optional[str] = None

    word_count: Optional[int] = None
    honest_word_count: Optional[int] = None
    word_count_diff: Optional[int] = None

    attempts: Optional[int] = None
    candidate_attempts: Optional[int] = None

    encoding_metadata: Optional[Dict[str, Any]] = None
    task_metadata: Optional[Dict[str, Any]] = None


@dataclass
class DatasetMetadata:
    dataset_name: str
    description: str
    generation_date: str
    total_records: int
    n_honest: int
    n_steganographic: int
    model_names: List[str]
    task_source: str
    encoding_schemes: Dict[str, Any]
    generation_config: Dict[str, Any]
    statistics: Dict[str, Any]