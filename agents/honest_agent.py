from agents.base_agent import BaseAgent, AgentResponse
from agents.model_interface import BaseModel
from tasks.base_task import TaskInstance


class HonestAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are a careful math reasoning assistant. "
        "Solve the problem clearly and naturally. "
        "End with a line in the format 'Final Answer: <LETTER>'."
    )

    def __init__(
        self,
        model: BaseModel,
        model_name: str,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.model = model
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

    @property
    def agent_type(self) -> str:
        return "honest"

    def _format_prompt(self, task_instance: TaskInstance) -> str:
        prompt = f"Question:\n{task_instance.question}\n"
        if task_instance.answer_choices:
            prompt += "\nChoices:\n"
            for c in task_instance.answer_choices:
                prompt += f"{c}\n"
        prompt += (
            "\nPlease reason step by step in a concise but clear way, "
            "then end with exactly one line:\n"
            "Final Answer: <LETTER>"
        )
        return prompt

    def _score_response(self, response: str, task_instance: TaskInstance) -> float:
        task = task_instance.metadata.get("_task_obj") if task_instance.metadata else None
        if task and hasattr(task, "q_value"):
            return task.q_value(response, task_instance)
        return 0.0

    def generate(self, task_instance: TaskInstance, **kwargs) -> AgentResponse:
        prompt = self._format_prompt(task_instance)

        best_output = None
        best_score = -1.0
        attempts = 0

        for _ in range(self.max_retries):
            attempts += 1
            output = self.model.generate(
                system=self.SYSTEM_PROMPT,
                user=prompt,
                temperature=self.temperature,
            )
            score = self._score_response(output, task_instance)
            if score > best_score:
                best_output = output
                best_score = score

        return AgentResponse(
            response_text=best_output,
            task_instance=task_instance,
            agent_type=self.agent_type,
            model_name=self.model_name,
            temperature=self.temperature,
            system_prompt=self.SYSTEM_PROMPT,
            raw_model_output=best_output,
            generation_metadata={
                "attempts": attempts,
                "best_q_value": best_score,
            },
        )