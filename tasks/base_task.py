from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaskInstance:
    question: str
    correct_answer: str
    answer_choices: Optional[List[str]] = None
    metadata: Optional[dict] = None


class BaseTask:
    def load_instances(self, n: int):
        raise NotImplementedError

    def is_correct(self, response: str, task_instance: TaskInstance) -> bool:
        return str(task_instance.correct_answer).lower() in response.lower()