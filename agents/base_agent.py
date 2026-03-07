from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from tasks.base_task import TaskInstance


@dataclass
class AgentResponse:
    response_text: str
    task_instance: TaskInstance
    agent_type: str
    model_name: str
    temperature: float
    system_prompt: str
    raw_model_output: Optional[str] = None
    generation_metadata: Optional[dict] = field(default_factory=dict)


class BaseAgent(ABC):
    @property
    @abstractmethod
    def agent_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, task_instance: TaskInstance, **kwargs) -> AgentResponse:
        raise NotImplementedError